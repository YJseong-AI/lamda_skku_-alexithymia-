import os
import cv2
import dlib
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pathlib import Path
from datetime import datetime
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
from collections import deque

app = Flask(__name__)
CORS(app)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent
shape_predictor_path = BASE_DIR / "shape_predictor_68_face_landmarks.dat"
emotion_model_path = BASE_DIR / "_mini_XCEPTION.102-0.66.hdf5"

# Dlib 초기화
print("[INFO] Initializing dlib...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(shape_predictor_path))
print("[OK] Dlib loaded")

# 감정 모델 로드
print("[INFO] Loading emotion model...")
emotion_classifier = load_model(str(emotion_model_path), compile=False)
print("[OK] Emotion model loaded")

# 감정 레이블
EMOTIONS_OLD = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
EMOTION_MAP = {
    "angry": "anger", "disgust": "disgust", "scared": "fear",
    "happy": "happy", "sad": "sad", "surprised": "surprise", "neutral": "neutral"
}

# 파라미터 (url.py와 동일)
WIN_SEC = 2.0
FPS_EST = 20
EMA_ALPHA = 0.3
MAD_MULTIPLIER = 2.5
FIXSTAB_ABS_THRESH = 0.30
CONSECUTIVE_UNSTABLE_THRESH = 3
EAR_THRESHOLD = 0.2

class EMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

def apply_clahe(image):
    """CLAHE 적용 - 대비 향상"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def calculate_ear(eye_points):
    """Eye Aspect Ratio 계산"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.3

def get_pupil_coords(eye_points):
    """동공 중심 좌표"""
    center = eye_points.mean(axis=0)
    return tuple(center.astype(int))

def calculate_fixation_stability(gaze_buffer):
    """Fixation Stability 계산"""
    if len(gaze_buffer) < 2:
        return None, None
    
    points = np.array(list(gaze_buffer))
    if points.shape[0] < 2:
        return None, None
    
    cov_matrix = np.cov(points.T)
    if cov_matrix.shape == ():
        cov_matrix = np.array([[cov_matrix]])
    elif cov_matrix.shape == (2,):
        cov_matrix = np.diag(cov_matrix)
    
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]
    
    if len(eigenvalues) < 2:
        eigenvalues = np.pad(eigenvalues, (0, 2-len(eigenvalues)), 'constant')
    
    lambda1, lambda2 = eigenvalues[0], eigenvalues[1]
    area = np.pi * np.sqrt(max(lambda1, 0)) * np.sqrt(max(lambda2, 0))
    fix_stab = 1 / (1 + area)
    
    return area, fix_stab

def calculate_mad(values):
    """중앙절대편차"""
    if len(values) == 0:
        return 0
    median = np.median(values)
    return np.median(np.abs(values - median))

# 세션 저장소
session_storage = {}

def process_frame(image_data, session_id='default'):
    """프레임 처리 - url.py 로직 완전 반영"""
    try:
        # 세션 초기화
        if session_id not in session_storage:
            session_storage[session_id] = {
                'gaze_buffer': deque(maxlen=int(WIN_SEC * FPS_EST)),
                'gaze_ema_x': EMA(EMA_ALPHA),
                'gaze_ema_y': EMA(EMA_ALPHA),
                'area_calibration_buffer': [],
                'calibration_done': False,
                'area_median': 0,
                'area_mad': 0,
                'consecutive_unstable_count': 0,
                'blink_count': 0,
                'was_blinking': False,
                'start_time': datetime.now(),
                'frame_count': 0
            }
        
        session = session_storage[session_id]
        session['frame_count'] += 1
        
        # Base64 디코딩
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape
        
        # CLAHE 적용
        gray = apply_clahe(frame)
        
        # Dlib 얼굴 감지
        faces = detector(gray)
        
        result = {
            "faces_detected": len(faces),
            "faces": [],
            "fixation_stability": None,
            "fixation_flag": 0,
            "blink_count": session['blink_count'],
            "elapsed_time": (datetime.now() - session['start_time']).total_seconds()
        }
        
        if len(faces) == 0:
            return result
        
        for face in faces:
            landmarks = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # 눈 랜드마크
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            # EAR 계산
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            is_blinking = avg_ear < EAR_THRESHOLD
            
            # 깜빡임 카운트
            if not session['was_blinking'] and is_blinking:
                session['blink_count'] += 1
            session['was_blinking'] = is_blinking
            result['blink_count'] = session['blink_count']
            
            # 동공 좌표
            left_pupil = get_pupil_coords(left_eye)
            right_pupil = get_pupil_coords(right_eye)
            
            # Fixation Stability 계산
            fix_stab = None
            fix_flag = 0
            
            if left_pupil and right_pupil:
                ipd = np.sqrt((left_pupil[0] - right_pupil[0])**2 + (left_pupil[1] - right_pupil[1])**2)
                
                if ipd > 0:
                    mx = (left_pupil[0] + right_pupil[0]) / 2
                    my = (left_pupil[1] + right_pupil[1]) / 2
                    
                    gaze_x_raw = mx / ipd
                    gaze_y_raw = my / ipd
                    
                    gaze_x = session['gaze_ema_x'].update(gaze_x_raw)
                    gaze_y = session['gaze_ema_y'].update(gaze_y_raw)
                    
                    session['gaze_buffer'].append((gaze_x, gaze_y))
                    
                    elapsed_time = (datetime.now() - session['start_time']).total_seconds()
                    
                    # 캘리브레이션 (처음 10초)
                    if elapsed_time <= 10 and not session['calibration_done']:
                        if len(session['gaze_buffer']) >= int(WIN_SEC * FPS_EST * 0.5):
                            area, _ = calculate_fixation_stability(session['gaze_buffer'])
                            if area is not None:
                                session['area_calibration_buffer'].append(area)
                    elif elapsed_time > 10 and not session['calibration_done']:
                        if session['area_calibration_buffer']:
                            session['area_median'] = np.median(session['area_calibration_buffer'])
                            session['area_mad'] = calculate_mad(np.array(session['area_calibration_buffer']))
                            session['calibration_done'] = True
                            print(f"Calibration done: Median={session['area_median']:.4f}, MAD={session['area_mad']:.4f}")
                    
                    # Fixation 계산
                    if len(session['gaze_buffer']) >= int(WIN_SEC * FPS_EST * 0.5):
                        fix_area, fix_stab = calculate_fixation_stability(session['gaze_buffer'])
                        
                        if fix_area is not None and fix_stab is not None:
                            result["fixation_stability"] = float(fix_stab)
                            
                            if session['calibration_done']:
                                unstable_thresh = session['area_median'] + MAD_MULTIPLIER * session['area_mad']
                                fix_flag = 1 if fix_area > unstable_thresh else 0
                            else:
                                fix_flag = 1 if fix_stab <= FIXSTAB_ABS_THRESH else 0
                            
                            result["fixation_flag"] = fix_flag
                            
                            if fix_flag == 1:
                                session['consecutive_unstable_count'] += 1
                            else:
                                session['consecutive_unstable_count'] = 0
                            
                            result["consecutive_unstable"] = session['consecutive_unstable_count']
                            result["distracted"] = session['consecutive_unstable_count'] >= CONSECUTIVE_UNSTABLE_THRESH
            
            face_data = {
                "bbox": {
                    "x": face.left(),
                    "y": face.top(),
                    "width": face.width(),
                    "height": face.height()
                },
                "left_pupil": {"x": int(left_pupil[0]), "y": int(left_pupil[1])},
                "right_pupil": {"x": int(right_pupil[0]), "y": int(right_pupil[1])},
                "is_blinking": bool(is_blinking),
                "ear": float(avg_ear)
            }
            
            # 감정 인식
            x, y, fw, fh = face.left(), face.top(), face.width(), face.height()
            roi = gray[y:y+fh, x:x+fw]
            if roi.size > 0:
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                
                preds = emotion_classifier.predict(roi, verbose=0)[0]
                emotion_label = EMOTIONS_OLD[preds.argmax()]
                
                face_data["emotion"] = EMOTION_MAP.get(emotion_label, emotion_label)
                face_data["emotion_scores"] = {
                    EMOTION_MAP.get(EMOTIONS_OLD[i], EMOTIONS_OLD[i]): float(preds[i])
                    for i in range(len(EMOTIONS_OLD))
                }
            
            result["faces"].append(face_data)
        
        return result
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_data = data.get('image')
    session_id = data.get('session_id', 'default')
    
    if not image_data:
        return jsonify({"error": "No image data"}), 400
    
    result = process_frame(image_data, session_id)
    return jsonify(result)

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session_id = request.json.get('session_id', 'default')
    if session_id in session_storage:
        del session_storage[session_id]
    return jsonify({"status": "ok"})

@app.route('/health')
def health():
    return jsonify({"status": "ok", "dlib": True, "emotion_model": True})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

