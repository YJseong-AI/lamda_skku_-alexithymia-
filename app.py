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

app = Flask(__name__)
CORS(app)

# 모델 경로 설정
BASE_DIR = Path(__file__).resolve().parent

# MediaPipe 초기화 (CPU 전용 모드)
os.environ['GLOG_minloglevel'] = '2'  # MediaPipe 로그 줄이기
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5,
    refine_landmarks=False  # GPU 사용 안 함
)
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0=short-range, 1=full-range (CPU 친화적)
    min_detection_confidence=0.5
)

# 감정 모델 로드
emotion_model_path = BASE_DIR / "_mini_XCEPTION.102-0.66.hdf5"
if emotion_model_path.exists():
    emotion_classifier = load_model(str(emotion_model_path), compile=False)
    print("Emotion model loaded successfully!")
else:
    print(f"Warning: Emotion model not found at {emotion_model_path}")
    emotion_classifier = None

EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
EMOTION_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "scared": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprised": "surprise",
    "neutral": "neutral"
}

def calculate_ear_mediapipe(landmarks, eye_indices):
    """MediaPipe 랜드마크로 EAR 계산"""
    # 눈 세로 거리
    vertical1 = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
    vertical2 = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
    # 눈 가로 거리
    horizontal = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
    
    if horizontal == 0:
        return 0.3
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def calculate_fixation_stability(gaze_points):
    """Fixation Stability 계산 (url.py와 동일)"""
    if len(gaze_points) < 2:
        return None, None
    
    points = np.array(gaze_points)
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

def process_frame(image_data, session_data=None):
    """프레임 처리 및 분석 (url.py 로직 반영)"""
    try:
        # Base64 이미지 디코딩
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w, _ = frame.shape
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # RGB로 변환 (MediaPipe용)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 감지
        detection_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)
        
        result = {
            "faces_detected": 0,
            "faces": [],
            "fixation_stability": None,
            "fixation_flag": 0
        }
        
        if detection_results.detections and mesh_results.multi_face_landmarks:
            for detection, face_landmarks in zip(detection_results.detections, mesh_results.multi_face_landmarks):
                bbox = detection.location_data.relative_bounding_box
                
                # 절대 좌표 변환
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                face_w = int(bbox.width * w)
                face_h = int(bbox.height * h)
                
                # 랜드마크 좌표 추출
                landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
                
                # 눈 인덱스 (MediaPipe Face Mesh)
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                
                # 동공 위치 (눈 중심)
                left_pupil = np.mean([landmarks[i] for i in LEFT_EYE], axis=0)
                right_pupil = np.mean([landmarks[i] for i in RIGHT_EYE], axis=0)
                
                # EAR 계산
                left_ear = calculate_ear_mediapipe(landmarks, LEFT_EYE)
                right_ear = calculate_ear_mediapipe(landmarks, RIGHT_EYE)
                avg_ear = (left_ear + right_ear) / 2.0
                is_blinking = avg_ear < 0.2
                
                # Fixation Stability 계산 (session_data에서 gaze_buffer 받음)
                fix_stab = None
                fix_flag = 0
                if session_data and 'gaze_buffer' in session_data:
                    # IPD 정규화된 시선 좌표 계산
                    ipd = np.sqrt((left_pupil[0] - right_pupil[0])**2 + (left_pupil[1] - right_pupil[1])**2)
                    if ipd > 0:
                        gaze_x = ((left_pupil[0] + right_pupil[0]) / 2) / ipd
                        gaze_y = ((left_pupil[1] + right_pupil[1]) / 2) / ipd
                        
                        gaze_buffer = session_data['gaze_buffer']
                        gaze_buffer.append((gaze_x, gaze_y))
                        
                        if len(gaze_buffer) >= 10:  # 최소 10프레임
                            area, fix_stab = calculate_fixation_stability(list(gaze_buffer))
                            if fix_stab is not None:
                                result["fixation_stability"] = float(fix_stab)
                                # 불안정 플래그 (임계값: 0.3)
                                fix_flag = 1 if fix_stab <= 0.3 else 0
                                result["fixation_flag"] = fix_flag
                
                face_data = {
                    "bbox": {
                        "x": max(0, x),
                        "y": max(0, y),
                        "width": face_w,
                        "height": face_h
                    },
                    "left_pupil": {"x": int(left_pupil[0]), "y": int(left_pupil[1])},
                    "right_pupil": {"x": int(right_pupil[0]), "y": int(right_pupil[1])},
                    "is_blinking": bool(is_blinking),
                    "ear": float(avg_ear)
                }
                
                # 감정 인식
                if emotion_classifier is not None and x >= 0 and y >= 0:
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
                        print(f"Emotion recognition error: {e}")
                
                result["faces"].append(face_data)
            
            result["faces_detected"] = len(result["faces"])
        
        return result
    
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

# 세션 데이터 저장 (간단한 메모리 기반)
from collections import deque as deque_collection
session_storage = {}

@app.route('/analyze', methods=['POST'])
def analyze():
    """프레임 분석 API (url.py 로직 반영)"""
    data = request.json
    image_data = data.get('image')
    session_id = data.get('session_id', 'default')
    
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400
    
    # 세션 데이터 초기화 또는 가져오기
    if session_id not in session_storage:
        session_storage[session_id] = {
            'gaze_buffer': deque_collection(maxlen=40),  # 2초 @ 20fps
            'blink_count': 0,
            'was_blinking': False
        }
    
    session_data = session_storage[session_id]
    
    # 프레임 처리
    result = process_frame(image_data, session_data)
    
    # 깜빡임 카운트 업데이트
    if result['faces_detected'] > 0:
        is_blinking = result['faces'][0]['is_blinking']
        if not session_data['was_blinking'] and is_blinking:
            session_data['blink_count'] += 1
        session_data['was_blinking'] = is_blinking
        result['blink_count'] = session_data['blink_count']
    
    return jsonify(result)

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """세션 초기화"""
    data = request.json
    session_id = data.get('session_id', 'default')
    if session_id in session_storage:
        del session_storage[session_id]
    return jsonify({"status": "ok"})

@app.route('/health')
def health():
    """헬스체크"""
    return jsonify({
        "status": "ok",
        "emotion_model_loaded": emotion_classifier is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

