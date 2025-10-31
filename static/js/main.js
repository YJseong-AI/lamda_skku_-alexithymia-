let video = document.getElementById('webcam');
let canvas = document.getElementById('overlay');
let ctx = canvas.getContext('2d');
let streaming = false;
let analyzeInterval = null;

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 1280, height: 720 } 
        });
        
        video.srcObject = stream;
        video.play();
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };
        
        streaming = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        document.getElementById('status').textContent = '✅ 카메라 실행 중 - 분석 진행중...';
        document.getElementById('status').classList.remove('error');
        
        // 프레임 분석 시작 (초당 5프레임)
        analyzeInterval = setInterval(analyzeFrame, 200);
        
    } catch (err) {
        console.error('카메라 접근 오류:', err);
        document.getElementById('status').textContent = '❌ 카메라 접근 실패: ' + err.message;
        document.getElementById('status').classList.add('error');
    }
}

function stopCamera() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }
    
    if (analyzeInterval) {
        clearInterval(analyzeInterval);
        analyzeInterval = null;
    }
    
    streaming = false;
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('status').textContent = '⏹️ 카메라 정지됨';
    
    // 캔버스 초기화
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('results').innerHTML = '';
}

async function analyzeFrame() {
    if (!streaming) return;
    
    // 비디오를 캔버스에 그리기
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0);
    
    // Base64로 인코딩
    const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        if (result.error) {
            console.error('분석 오류:', result.error);
            return;
        }
        
        // 결과 표시
        displayResults(result);
        drawOverlay(result);
        
    } catch (err) {
        console.error('분석 요청 오류:', err);
    }
}

function drawOverlay(result) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (result.faces_detected === 0) return;
    
    result.faces.forEach(face => {
        // 얼굴 박스
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 3;
        ctx.strokeRect(face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height);
        
        // 동공 위치
        ctx.fillStyle = '#ff0000';
        ctx.beginPath();
        ctx.arc(face.left_pupil.x, face.left_pupil.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(face.right_pupil.x, face.right_pupil.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        // 감정 레이블
        if (face.emotion) {
            ctx.fillStyle = '#00ff00';
            ctx.font = 'bold 24px Arial';
            ctx.fillText(face.emotion.toUpperCase(), face.bbox.x, face.bbox.y - 10);
        }
        
        // 깜빡임 표시
        if (face.is_blinking) {
            ctx.fillStyle = '#ff0000';
            ctx.font = 'bold 20px Arial';
            ctx.fillText('BLINKING', face.bbox.x, face.bbox.y + face.bbox.height + 25);
        }
    });
}

function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    
    if (result.faces_detected === 0) {
        resultsDiv.innerHTML = '<div class="result-card"><h3>얼굴이 감지되지 않았습니다</h3></div>';
        return;
    }
    
    let html = '';
    
    result.faces.forEach((face, index) => {
        html += `
            <div class="result-card">
                <h3>👤 얼굴 #${index + 1}</h3>
                
                ${face.emotion ? `
                    <div style="margin-bottom: 20px;">
                        <h4 style="color: #667eea; margin-bottom: 10px;">🎭 감정: ${face.emotion.toUpperCase()}</h4>
                        ${Object.entries(face.emotion_scores || {}).map(([emotion, score]) => `
                            <div class="emotion-bar">
                                <div class="emotion-label">
                                    <span>${emotion}</span>
                                    <span>${(score * 100).toFixed(1)}%</span>
                                </div>
                                <div class="bar-container">
                                    <div class="bar-fill" style="width: ${score * 100}%"></div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                ` : ''}
                
                <h4 style="color: #667eea; margin-bottom: 10px;">👁️ 시선 정보</h4>
                <div class="info-item">
                    <span>왼쪽 동공</span>
                    <span>(${face.left_pupil.x}, ${face.left_pupil.y})</span>
                </div>
                <div class="info-item">
                    <span>오른쪽 동공</span>
                    <span>(${face.right_pupil.x}, ${face.right_pupil.y})</span>
                </div>
                <div class="info-item">
                    <span>깜빡임</span>
                    <span>${face.is_blinking ? '😑 감은 상태' : '👁️ 뜬 상태'}</span>
                </div>
                <div class="info-item">
                    <span>EAR 값</span>
                    <span>${face.ear.toFixed(3)}</span>
                </div>
            </div>
        `;
    });
    
    resultsDiv.innerHTML = html;
}

// 페이지 로드 시 상태 확인
fetch('/health')
    .then(res => res.json())
    .then(data => {
        if (!data.emotion_model_loaded) {
            document.getElementById('status').textContent = '⚠️ 감정 인식 모델이 로드되지 않았습니다 (시선 추적만 가능)';
        }
    })
    .catch(err => console.error('Health check failed:', err));

