# Google Cloud Run 배포 가이드

## 1. 사전 준비

### Google Cloud SDK 설치
```bash
# Windows (PowerShell)
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")
& $env:Temp\GoogleCloudSDKInstaller.exe
```

### 로그인 및 프로젝트 설정
```bash
gcloud init
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

## 2. 로컬 테스트

### Docker 빌드
```bash
docker build -f Dockerfile.cloudrun -t gazefusion-pro .
```

### 로컬 실행
```bash
docker run -p 8080:8080 gazefusion-pro
```

브라우저에서 `http://localhost:8080` 접속

## 3. Cloud Run 배포

### 방법 1: gcloud CLI (권장)
```bash
# 프로젝트 ID 설정
gcloud config set project YOUR_PROJECT_ID

# Cloud Build API 활성화
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# 배포
gcloud run deploy gazefusion-pro \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --max-instances 10 \
  --min-instances 0
```

### 방법 2: Docker 이미지 직접 빌드 후 배포
```bash
# Artifact Registry 저장소 생성
gcloud artifacts repositories create gazefusion \
  --repository-format=docker \
  --location=us-central1

# Docker 인증
gcloud auth configure-docker us-central1-docker.pkg.dev

# 이미지 빌드 및 푸시
docker build -f Dockerfile.cloudrun -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/gazefusion/gazefusion-pro:latest .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/gazefusion/gazefusion-pro:latest

# Cloud Run 배포
gcloud run deploy gazefusion-pro \
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/gazefusion/gazefusion-pro:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

## 4. 배포 후 확인

배포가 완료되면 URL이 표시됩니다:
```
Service URL: https://gazefusion-pro-xxxxx-uc.a.run.app
```

해당 URL로 접속하여 테스트하세요!

## 5. 비용

**무료 티어 (매월):**
- 2백만 요청
- 360,000 vCPU-초
- 180,000 GiB-초 메모리

**예상 비용:**
- 소규모 사용: 무료
- 중규모 사용: $5-20/월

## 6. 업데이트

코드 수정 후 재배포:
```bash
gcloud run deploy gazefusion-pro \
  --source . \
  --platform managed \
  --region us-central1
```

## 7. 로그 확인

```bash
gcloud run logs read gazefusion-pro
```

## 8. 삭제

```bash
gcloud run services delete gazefusion-pro --region us-central1
```

## 문제 해결

### 메모리 부족
```bash
gcloud run services update gazefusion-pro --memory 8Gi
```

### 타임아웃
```bash
gcloud run services update gazefusion-pro --timeout 600
```

### Cold Start 개선
```bash
gcloud run services update gazefusion-pro --min-instances 1
```

