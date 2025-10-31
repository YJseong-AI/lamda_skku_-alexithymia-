# GazeFusion Pro - Web Application

실시간 감정 인식 및 시선 추적 웹 애플리케이션

## 기능

- 🎭 실시간 감정 인식 (7가지 감정)
- 👁️ 시선 추적 (동공 위치 감지)
- 😑 깜빡임 감지
- 🌐 웹 기반 (브라우저에서 실행)

## 로컬 실행

### 1. 가상환경 설정
```bash
conda create -n gazefusion python=3.11
conda activate gazefusion
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 모델 파일 준비
프로젝트 루트에 `_mini_XCEPTION.102-0.66.hdf5` 파일 배치
(없으면 시선 추적만 작동)

### 4. 실행
```bash
python app.py
```

브라우저에서 `http://localhost:5000` 접속

## Railway 배포

### 1. GitHub에 Push
```bash
git add .
git commit -m "Deploy to Railway"
git push origin master
```

### 2. Railway 설정
1. Railway.app 접속
2. "New Project" → "Deploy from GitHub repo"
3. Repository 선택: `lamda_skku_-alexithymia-`
4. 자동 배포 시작

### 3. 환경 변수 (선택사항)
Railway 대시보드에서 설정:
- `PORT`: 자동 설정됨
- `PYTHON_VERSION`: 3.11.6

## 주의사항

⚠️ **모델 파일 크기**
- `_mini_XCEPTION.102-0.66.hdf5` (약 2MB) - 감정 인식용
- `shape_predictor_68_face_landmarks.dat` (95MB) - 자동 다운로드

Railway 무료 플랜은 저장 공간 제한이 있으므로, 큰 모델 파일은 자동 다운로드 방식 사용

## 기술 스택

- **Backend**: Flask
- **Frontend**: HTML5, JavaScript, WebRTC
- **ML Models**: 
  - Dlib (얼굴 랜드마크)
  - Keras/TensorFlow (감정 인식)

## 라이선스

MIT License

