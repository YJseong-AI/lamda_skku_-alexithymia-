import os
import cv2
import dlib
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pathlib import Path
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import urllib.request
import bz2

app = Flask(__name__)
CORS(app)

# 모델 경로 설정
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Shape Predictor 자동 다운로드
shape_predictor_path = MODEL_DIR / "shape_predictor_68_face_landmarks.dat"
if not shape_predictor_path.exists():
    print("Downloading dlib face landmark model...")
    url = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = MODEL_DIR / "shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        urllib.request.urlretrieve(url, str(compressed_file))
        with bz2.open(str(compressed_file), 'rb') as f_in:
            with open(str(shape_predictor_path), 'wb') as f_out:
                f_out.write(f_in.read())
        compressed_file.unlink()
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")

# Dlib 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(str(shape_predictor_path))

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

def calculate_ear(eye_points):
    """Eye Aspect Ratio 계산"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_pupil_coords(eye_points):
    """동공 중심 좌표 계산"""
    center = eye_points.mean(axis=0)
    return tuple(center.astype(int))

def process_frame(image_data):
    """프레임 처리 및 분석"""
    try:
        # Base64 이미지 디코딩
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지
        faces = detector(gray)
        
        result = {
            "faces_detected": len(faces),
            "faces": []
        }
        
        for face in faces:
            landmarks = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # 눈 랜드마크
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            # EAR 계산 (깜빡임)
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            is_blinking = avg_ear < 0.2
            
            # 동공 좌표
            left_pupil = get_pupil_coords(left_eye)
            right_pupil = get_pupil_coords(right_eye)
            
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
            if emotion_classifier is not None:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                roi = gray[y:y+h, x:x+w]
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
            
            result["faces"].append(face_data)
        
        return result
    
    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """프레임 분석 API"""
    data = request.json
    image_data = data.get('image')
    
    if not image_data:
        return jsonify({"error": "No image data provided"}), 400
    
    result = process_frame(image_data)
    return jsonify(result)

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

