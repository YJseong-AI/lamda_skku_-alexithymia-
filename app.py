import os
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pathlib import Path
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
from collections import deque as deque_collection

app = Flask(__name__)
CORS(app)

# 모델 경로 설정
BASE_DIR = Path(__file__).resolve().parent

# MediaPipe 초기화 (민감도 높임)
os.environ['GLOG_minloglevel'] = '2'
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.2,  # 매우 민감하게
    min_tracking_confidence=0.2
)

# 감정 모델 로드
emotion_model_path = BASE_DIR / "_mini_XCEPTION.102-0.66.hdf5"
if emotion_model_path.exists():
    emotion_classifier = load_model(str(emotion_model_path), compile=False)
    print("[OK] Emotion model loaded!")
else:
    print(f"[WARN] Emotion model not found")
    emotion_classifier = None

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
EMOTION_MAP = {
    "angry": "anger", "disgust": "disgust", "scared": "fear",
    "happy": "happy", "sad": "sad", "surprised": "surprise", "neutral": "neutral"
}

def apply_clahe(image):
    """CLAHE 적용 - url.py와 동일"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def calculate_ear(landmarks, eye_indices):
    """EAR 계산"""
    v1 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
    v2 = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
    h = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
    return (v1 + v2) / (2.0 * h) if h > 0 else 0.3

def calculate_fixation_stability(gaze_points):
    """Fixation Stability 계산"""
    if len(gaze_points) < 2:
        return None, None
    
    points = np.array(gaze_points)
    cov_matrix = np.cov(points.T)
    if cov_matrix.shape == ():
        cov_matrix = np.array([[cov_matrix]])
    elif cov_matrix.shape == (2,):
        cov_matrix = np.diag(cov_matrix)
    
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues = np.real(np.sort(eigenvalues)[::-1])
    if len(eigenvalues) < 2:
        eigenvalues = np.pad(eigenvalues, (0, 2-len(eigenvalues)), 'constant')
    
    lambda1, lambda2 = eigenvalues[0], eigenvalues[1]
    area = np.pi * np.sqrt(max(lambda1, 0)) * np.sqrt(max(lambda2, 0))
    fix_stab = 1 / (1 + area)
    
    return area, fix_stab

def process_frame(image_data, session_data=None):
    """프레임 처리 (url.py 로직)"""
    try:
        # Base64 디코딩
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape
        
        # CLAHE 적용
        gray = apply_clahe(frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Face Mesh
        mesh_results = face_mesh.process(rgb_frame)
        
        result = {
            "faces_detected": 0,
            "faces": [],
            "fixation_stability": None,
            "fixation_flag": 0
        }
        
        if not mesh_results.multi_face_landmarks:
            return result
        
        for face_landmarks in mesh_results.multi_face_landmarks:
            # 랜드마크 좌표
            landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
            
            # Bounding Box 계산
            xs = [coord[0] for coord in landmarks]
            ys = [coord[1] for coord in landmarks]
            x, y = int(min(xs)), int(min(ys))
            face_w, face_h = int(max(xs) - min(xs)), int(max(ys) - min(ys))
            
            # 눈 인덱스
            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]
            
            # 동공 위치
            left_pupil = np.mean([landmarks[i] for i in LEFT_EYE], axis=0)
            right_pupil = np.mean([landmarks[i] for i in RIGHT_EYE], axis=0)
            
            # EAR 계산
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            is_blinking = avg_ear < 0.2
            
            # Fixation Stability
            fix_stab = None
            fix_flag = 0
            if session_data and 'gaze_buffer' in session_data:
                ipd = np.sqrt((left_pupil[0] - right_pupil[0])**2 + (left_pupil[1] - right_pupil[1])**2)
                if ipd > 0:
                    gaze_x = ((left_pupil[0] + right_pupil[0]) / 2) / ipd
                    gaze_y = ((left_pupil[1] + right_pupil[1]) / 2) / ipd
                    
                    session_data['gaze_buffer'].append((gaze_x, gaze_y))
                    
                    if len(session_data['gaze_buffer']) >= 10:
                        area, fix_stab = calculate_fixation_stability(list(session_data['gaze_buffer']))
                        if fix_stab is not None:
                            result["fixation_stability"] = float(fix_stab)
                            fix_flag = 1 if fix_stab <= 0.3 else 0
                            result["fixation_flag"] = fix_flag
            
            face_data = {
                "bbox": {"x": max(0, x), "y": max(0, y), "width": face_w, "height": face_h},
                "left_pupil": {"x": int(left_pupil[0]), "y": int(left_pupil[1])},
                "right_pupil": {"x": int(right_pupil[0]), "y": int(right_pupil[1])},
                "is_blinking": bool(is_blinking),
                "ear": float(avg_ear)
            }
            
            # 감정 인식
            if emotion_classifier and x >= 0 and y >= 0:
                try:
                    roi = gray[y:y+face_h, x:x+face_w]
                    if roi.size > 0:
                        roi = cv2.resize(roi, (64, 64))
                        roi = roi.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)
                        
                        preds = emotion_classifier.predict(roi, verbose=0)[0]
                        emotion_label = EMOTIONS[preds.argmax()]
                        
                        face_data["emotion"] = EMOTION_MAP.get(emotion_label, emotion_label)
                        face_data["emotion_scores"] = {
                            EMOTION_MAP.get(EMOTIONS[i], EMOTIONS[i]): float(preds[i])
                            for i in range(len(EMOTIONS))
                        }
                except Exception as e:
                    print(f"Emotion error: {e}")
            
            result["faces"].append(face_data)
        
        result["faces_detected"] = len(result["faces"])
        return result
    
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

# 세션 저장소
session_storage = {}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_data = data.get('image')
    session_id = data.get('session_id', 'default')
    
    if not image_data:
        return jsonify({"error": "No image data"}), 400
    
    if session_id not in session_storage:
        session_storage[session_id] = {
            'gaze_buffer': deque_collection(maxlen=40),
            'blink_count': 0,
            'was_blinking': False
        }
    
    session_data = session_storage[session_id]
    result = process_frame(image_data, session_data)
    
    if result['faces_detected'] > 0:
        is_blinking = result['faces'][0]['is_blinking']
        if not session_data['was_blinking'] and is_blinking:
            session_data['blink_count'] += 1
        session_data['was_blinking'] = is_blinking
        result['blink_count'] = session_data['blink_count']
    
    return jsonify(result)

@app.route('/reset_session', methods=['POST'])
def reset_session():
    session_id = request.json.get('session_id', 'default')
    if session_id in session_storage:
        del session_storage[session_id]
    return jsonify({"status": "ok"})

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "emotion_model_loaded": emotion_classifier is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
