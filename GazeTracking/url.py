import os
import cv2
import dlib
import ctypes
from pathlib import Path
from datetime import datetime
import pandas as pd
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import bz2
import urllib.request
import time
import csv

# ========== 모델 파일 경로 설정 ==========
script_dir = Path(__file__).resolve().parent
# 프로젝트 루트에 있는 파일 사용
model_path = script_dir.parent / "shape_predictor_68_face_landmarks.dat"

# 파일 존재 확인
if model_path.exists():
    file_size = model_path.stat().st_size
    print(f"[OK] 모델 파일 확인됨: {model_path}")
    print(f"  파일 크기: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
else:
    print(f"[ERROR] 모델 파일이 없습니다: {model_path}")
    exit(1)

# ——— 기본 설정 ———
SKIP_FRAMES = 5
MAX_WORKERS = 1
PRED_BUFFER_SIZE = 3
PUPIL_BUFFER_SIZE = 3

# ===== Fixation 관련 파라미터 =====
WIN_SEC = 2.0
HOP_SEC = 1.0
EMA_ALPHA = 0.3
FPS_EST = 20
CALIBRATION_SEC = 10
MAD_MULTIPLIER = 2.5
FIXSTAB_ABS_THRESH = 0.30
CONSECUTIVE_UNSTABLE_THRESH = 3

# DPI 스케일 설정
ctypes.windll.shcore.SetProcessDpiAwareness(1)
hDC = ctypes.windll.user32.GetDC(0)
dpi = ctypes.windll.gdi32.GetDeviceCaps(hDC, 88)
ctypes.windll.user32.ReleaseDC(0, hDC)
scale_factor = dpi / 96.0

# ===== EMA 클래스 =====
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
    """중앙절대편차(MAD) 계산"""
    if len(values) == 0:
        return 0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return mad

# 캡처-처리 버퍼
frame_queue = queue.Queue(maxsize=1)

# 스무딩 버퍼
preds_buffer = deque(maxlen=PRED_BUFFER_SIZE)
label_buffer = deque(maxlen=PRED_BUFFER_SIZE)
lp_buffer = deque(maxlen=PUPIL_BUFFER_SIZE)
rp_buffer = deque(maxlen=PUPIL_BUFFER_SIZE)

# Fixation 관련 변수
gaze_buffer = deque(maxlen=int(WIN_SEC * FPS_EST))
gaze_ema_x = EMA(EMA_ALPHA)
gaze_ema_y = EMA(EMA_ALPHA)
area_calibration_buffer = []
calibration_done = False
area_median = 0
area_mad = 0
consecutive_unstable_count = 0

# 워커용 ThreadPool
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
pending_future = None
last_label, last_preds = None, np.zeros(7)

# 버튼 입력 전역 변수
button_pressed = False
button_press_time = 0

# 깜빡임 카운트 변수
blink_count = 0
was_blinking = False

# 감정 레이블 (첫 번째 코드 순서)
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
EMOTIONS_OLD = ["angry","disgust","scared","happy","sad","surprised","neutral"]

# 추론 작업 함수
def inference_task(roi_gray):
    roi = cv2.resize(roi_gray, (64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_classifier.predict(roi, verbose=0)[0]
    return preds

# 캡처 루프
def capture_loop():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(frame)

# 초기화
# user_name = input("Please Enter Your Name: ")
user_name = "User"  # 기본 사용자 이름 (비대화형 모드)

# 모델 및 탐지기 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 20)

print("\nGazeTracking 초기화 중...")

# Dlib 얼굴 탐지기 및 랜드마크 예측기
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(str(model_path))
    print("[OK] Dlib 랜드마크 예측기 로드 완료")
except Exception as e:
    print(f"[ERROR] Dlib 랜드마크 예측기 로드 실패: {e}")
    exit(1)

# 감정 모델 로드 (상위 폴더에 위치)
emotion_model_path = script_dir.parent / "_mini_XCEPTION.102-0.66.hdf5"

if not emotion_model_path.exists():
    print(f"[ERROR] 감정 모델을 찾을 수 없습니다: {emotion_model_path}")
    exit(1)

emotion_classifier = load_model(str(emotion_model_path), compile=False)
print("[OK] 감정 분류 모델 로드 완료")

# 비디오 저장 설정
base_dir = script_dir / "data"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = base_dir / current_time
output_dir.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 20.0
frame_width, frame_height = 1280, 720
out = cv2.VideoWriter(str(output_dir / "output.mp4"), fourcc, fps, (frame_width, frame_height))

# CSV 초기화
csv_filename = output_dir / f"{user_name}_log.csv"
base_columns = ['Frame', 'Time(s)'] + emotions + ['Emotion', 'Left_Pupil_X', 'Left_Pupil_Y', 
                'Right_Pupil_X', 'Right_Pupil_Y', 'Blink', 'Blink_Count', 'Button_Press', 'Button_Time']
new_columns = ['FixStab', 'FixFlag']

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(base_columns + new_columns)

# 감정 히스토리
emotion_history = deque(maxlen=30)

# CLAHE 적용 함수
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

# EAR 계산 함수
def calculate_ear(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 동공 중심 계산 함수
def get_pupil_coords(eye_points):
    center = eye_points.mean(axis=0)
    return tuple(center.astype(int))

# 캡처 스레드 시작
threading.Thread(target=capture_loop, daemon=True).start()

print("\n카메라가 시작됩니다. Enter 키를 누르면 사용자 이벤트가 기록됩니다.")
print("종료하려면 Q 키를 누르세요.")

# 폰트 설정 (첫 번째 코드 스타일)
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_color = (0, 255, 0)
thickness = 3
line_type = cv2.LINE_AA

# 메인 루프
frame_count = 0
start_time = time.time()
last_logged_time = time.time()
EAR_THRESHOLD = 0.2

while True:
    if frame_queue.empty():
        continue
    
    frame = frame_queue.get()
    frame = cv2.resize(frame, (frame_width, frame_height))
    h, w, _ = frame.shape

    # 키 입력 확인 (waitKey를 메인 루프 시작 부분에서 호출)
    key = cv2.waitKey(1) & 0xFF
    
    # Enter 키 감지 (Windows에서 제대로 작동)
    if key == 13:  # Enter 키
        button_pressed = True
        button_press_time = round(time.time() - start_time, 3)
        print(f"[Button Pressed] Time: {button_press_time}s")
    else:
        button_pressed = False

    sm_lp = None
    sm_rp = None
    blink_flag = 0
    fix_stab = None
    fix_flag = 0

    gray = apply_clahe(frame)
    faces = detector(gray)

    # 감정 추론 스케줄링
    if frame_count % SKIP_FRAMES == 0 and faces:
        x0, y0, w0, h0 = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
        roi_gray = gray[y0:y0+h0, x0:x0+w0]
        if not pending_future or pending_future.done():
            pending_future = executor.submit(inference_task, roi_gray)

    # 추론 완료 후 스무딩
    if pending_future and pending_future.done():
        preds = pending_future.result()
        preds_buffer.append(preds)
        avg_preds = np.mean(preds_buffer, axis=0)
        label_buffer.append(EMOTIONS_OLD[avg_preds.argmax()])
        last_preds = avg_preds
        last_label = max(label_buffer, key=label_buffer.count)
        pending_future = None

    # 얼굴 탐지 및 분석
    if faces:
        for face in faces:
            landmarks = predictor(gray, face)
            x, y, fw, fh = face.left(), face.top(), face.width(), face.height()
            
            # 랜드마크를 numpy 배열로 변환
            shape = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # 왼쪽 눈 (36-41), 오른쪽 눈 (42-47)
            left_eye = shape[36:42]
            right_eye = shape[42:48]
            
            # 눈 랜드마크 그리기 (첫 번째 코드 스타일)
            for pt in left_eye:
                cv2.circle(frame, tuple(pt), 2, (0, 255, 255), -1)
            for pt in right_eye:
                cv2.circle(frame, tuple(pt), 2, (0, 255, 255), -1)
            
            # EAR 계산 (깜빡임 탐지)
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            is_blinking = avg_ear < EAR_THRESHOLD
            
            # 깜빡임 카운트 (첫 번째 코드 로직)
            if not was_blinking and is_blinking:
                blink_count += 1
                blink_text = f"Blinking ({blink_count})"
                cv2.putText(frame, blink_text, (10, 30), font, 1, (0, 0, 255), 2)
            
            was_blinking = is_blinking
            blink_flag = 1 if is_blinking else 0
            
            # 동공 좌표 추정
            lp = get_pupil_coords(left_eye)
            rp = get_pupil_coords(right_eye)
            
            if lp:
                lp_buffer.append(lp)
            if rp:
                rp_buffer.append(rp)
            
            sm_lp = tuple(map(int, np.mean(lp_buffer, axis=0))) if lp_buffer else lp
            sm_rp = tuple(map(int, np.mean(rp_buffer, axis=0))) if rp_buffer else rp
            
            # ===== Fixation Stability 계산 =====
            if sm_lp and sm_rp:
                # IPD
                ipd = np.sqrt((sm_lp[0] - sm_rp[0])**2 + (sm_lp[1] - sm_rp[1])**2)
                
                # 시선 중점
                mx = (sm_lp[0] + sm_rp[0]) / 2
                my = (sm_lp[1] + sm_rp[1]) / 2
                
                # IPD 정규화
                if ipd > 0:
                    gaze_x_raw = mx / ipd
                    gaze_y_raw = my / ipd
                    
                    # EMA 평활화
                    gaze_x = gaze_ema_x.update(gaze_x_raw)
                    gaze_y = gaze_ema_y.update(gaze_y_raw)
                    
                    # 시선 버퍼에 추가
                    gaze_buffer.append((gaze_x, gaze_y))
            
            elapsed_time = time.time() - start_time
            
            # 캘리브레이션
            if elapsed_time <= CALIBRATION_SEC and not calibration_done:
                if len(gaze_buffer) >= int(WIN_SEC * FPS_EST * 0.5):
                    area, _ = calculate_fixation_stability(gaze_buffer)
                    if area is not None:
                        area_calibration_buffer.append(area)
            elif elapsed_time > CALIBRATION_SEC and not calibration_done:
                if area_calibration_buffer:
                    area_median = np.median(area_calibration_buffer)
                    area_mad = calculate_mad(np.array(area_calibration_buffer))
                    calibration_done = True
                    print(f"Calibration completed: Median={area_median:.4f}, MAD={area_mad:.4f}")
            
            # Fixation 계산
            if len(gaze_buffer) >= int(WIN_SEC * FPS_EST * 0.5):
                fix_area, fix_stab = calculate_fixation_stability(gaze_buffer)
                
                if fix_area is not None and fix_stab is not None:
                    if calibration_done:
                        unstable_thresh = area_median + MAD_MULTIPLIER * area_mad
                        fix_flag = 1 if fix_area > unstable_thresh else 0
                    else:
                        fix_flag = 1 if fix_stab <= FIXSTAB_ABS_THRESH else 0
                    
                    if fix_flag == 1:
                        consecutive_unstable_count += 1
                    else:
                        consecutive_unstable_count = 0
            
            # 얼굴 박스 그리기
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            
            # 감정 표시 (첫 번째 코드 스타일)
            if last_label:
                # 감정 레이블 매핑 (old -> new)
                emotion_map = {
                    "angry": "anger",
                    "disgust": "disgust",
                    "scared": "fear",
                    "happy": "happy",
                    "sad": "sad",
                    "surprised": "surprise",
                    "neutral": "neutral"
                }
                display_emotion = emotion_map.get(last_label, last_label)
                emotion_history.append(display_emotion)
                
                cv2.putText(frame, display_emotion, (x, y - 10), font, font_scale, 
                           font_color, thickness, line_type)
            
            # 감정 확률 표시 (첫 번째 코드 스타일)
            y_offset = y + fh + 20
            for idx, emo in enumerate(emotions):
                # 확률 매핑
                old_idx_map = {
                    'happy': EMOTIONS_OLD.index('happy'),
                    'surprise': EMOTIONS_OLD.index('surprised'),
                    'sad': EMOTIONS_OLD.index('sad'),
                    'anger': EMOTIONS_OLD.index('angry'),
                    'disgust': EMOTIONS_OLD.index('disgust'),
                    'fear': EMOTIONS_OLD.index('scared'),
                    'neutral': EMOTIONS_OLD.index('neutral')
                }
                score = last_preds[old_idx_map[emo]]
                txt = f"{emo}: {score:.2f}"
                cv2.putText(frame, txt, (x, y_offset + idx * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            
            # 감정 히스토리 표시
            for i, emo in enumerate(list(emotion_history)[-5:]):
                cv2.putText(frame, f"History-{i+1}: {emo}", (10, 300 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 버튼이 눌렸을 때 화면에 표시
    if button_pressed:
        cv2.putText(frame, "USER EVENT!", (w//2 - 100, 50), font, 1.5, (0, 0, 255), 3)
        cv2.circle(frame, (50, 50), 20, (0, 0, 255), -1)

    # Fixation 정보 표시
    if fix_stab is not None:
        fixation_text = f"FixStab: {fix_stab:.3f} | FixFlag: {fix_flag}"
        cv2.putText(frame, fixation_text, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if consecutive_unstable_count >= CONSECUTIVE_UNSTABLE_THRESH:
        cv2.putText(frame, "DISTRACTED SEGMENT!", (10, 480), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 캘리브레이션 상태 표시
    elapsed_time = time.time() - start_time
    if not calibration_done:
        calib_time_left = max(0, CALIBRATION_SEC - elapsed_time)
        cv2.putText(frame, f"Calibrating... {calib_time_left:.1f}s", 
                   (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    # 사용법 안내
    cv2.putText(frame, "Press ENTER for User Event | Press Q to Quit", 
               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # CSV 데이터 저장
    current_time = time.time()
    elapsed_time_display = round(current_time - start_time, 3)
    
    if current_time - last_logged_time >= 1 or button_pressed:
        # 감정 확률값 매핑
        emotion_probs = []
        for emo in emotions:
            old_idx_map = {
                'happy': EMOTIONS_OLD.index('happy'),
                'surprise': EMOTIONS_OLD.index('surprised'),
                'sad': EMOTIONS_OLD.index('sad'),
                'anger': EMOTIONS_OLD.index('angry'),
                'disgust': EMOTIONS_OLD.index('disgust'),
                'fear': EMOTIONS_OLD.index('scared'),
                'neutral': EMOTIONS_OLD.index('neutral')
            }
            emotion_probs.append(float(last_preds[old_idx_map[emo]]))
        
        # 현재 감정 레이블 변환
        emotion_map = {
            "angry": "anger", "disgust": "disgust", "scared": "fear",
            "happy": "happy", "sad": "sad", "surprised": "surprise", "neutral": "neutral"
        }
        current_emotion = emotion_map.get(last_label, last_label) if last_label else None
        
        base_data = [
            frame_count,
            elapsed_time_display
        ] + emotion_probs + [
            current_emotion,
            sm_lp[0] if sm_lp else None, 
            sm_lp[1] if sm_lp else None,
            sm_rp[0] if sm_rp else None, 
            sm_rp[1] if sm_rp else None,
            int(blink_flag),
            int(blink_count),
            int(button_pressed),
            button_press_time if button_pressed else 0
        ]
        
        new_data = [
            fix_stab if fix_stab is not None else '',
            fix_flag
        ]
        
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(base_data + new_data)
            
            if button_pressed:
                print(f"데이터 저장됨 - Frame: {frame_count}, 감정: {current_emotion}, 시간: {elapsed_time_display}s")
        
        last_logged_time = current_time

    frame_count += 1
    out.write(frame)
    cv2.imshow("Emotion Estimation & Eye Tracker", frame)
    
    if key == ord("q"):
        break

# 종료 정리
cap.release()
out.release()
cv2.destroyAllWindows()
executor.shutdown(wait=True)

print(f"\n실험 완료! 데이터가 저장되었습니다: {csv_filename}")
if calibration_done:
    print(f"Calibration results - Median: {area_median:.4f}, MAD: {area_mad:.4f}")
else:
    print("Calibration was not completed (insufficient data)")