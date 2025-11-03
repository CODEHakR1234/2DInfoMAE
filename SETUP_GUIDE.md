# 🔧 InfoMAE 환경 설정 가이드

완벽한 개발 환경을 구축하기 위한 단계별 가이드입니다.

---

## 📋 선택지

두 가지 방법 중 선택하세요:

1. **Python venv** (가볍고 빠름, 권장)
2. **Conda** (패키지 관리가 편리함)

---

## 🐍 Option 1: Python venv (권장)

### 1단계: 사전 요구사항 확인

```bash
# Python 버전 확인 (3.8 이상 필요)
python3 --version

# pip 확인
python3 -m pip --version
```

### 2단계: 자동 설치 실행

```bash
cd /Users/ihagmyeong/Documents/VClab/2DInfoMAE

# 실행 권한 부여 (이미 완료됨)
chmod +x setup_venv.sh

# 자동 설치 실행
bash setup_venv.sh
```

### 3단계: 환경 활성화

```bash
# 활성화
source venv/bin/activate

# 프롬프트가 (venv)로 시작하면 성공!
```

### 4단계: 설치 검증

```bash
python test_installation.py
```

### 일상적인 사용

```bash
# 작업 시작할 때
cd /Users/ihagmyeong/Documents/VClab/2DInfoMAE
source venv/bin/activate

# 작업 끝날 때
deactivate
```

---

## 🐍 Option 2: Conda

### 1단계: Conda 설치 확인

```bash
# Conda 버전 확인
conda --version

# 없다면 설치: https://docs.conda.io/en/latest/miniconda.html
```

### 2단계: 자동 설치 실행

```bash
cd /Users/ihagmyeong/Documents/VClab/2DInfoMAE

# 실행 권한 부여 (이미 완료됨)
chmod +x setup_conda.sh

# 자동 설치 실행
bash setup_conda.sh
```

### 3단계: 환경 활성화

```bash
# 활성화
conda activate infomae

# 프롬프트가 (infomae)로 시작하면 성공!
```

### 4단계: 설치 검증

```bash
python test_installation.py
```

### 일상적인 사용

```bash
# 작업 시작할 때
conda activate infomae

# 작업 끝날 때
conda deactivate
```

---

## 🔧 수동 설치 (문제 발생시)

### Python venv 수동 설치

```bash
# 1. 가상환경 생성
python3 -m venv venv

# 2. 활성화
source venv/bin/activate

# 3. pip 업그레이드
pip install --upgrade pip

# 4. PyTorch 설치 (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 또는 CPU만
pip install torch torchvision

# 5. 나머지 패키지 설치
pip install timm numpy pillow matplotlib seaborn scikit-learn scipy
pip install tensorboard tqdm einops wandb opencv-python pandas

# 6. 설치 확인
python test_installation.py
```

### Conda 수동 설치

```bash
# 1. 환경 생성
conda create -n infomae python=3.9 -y

# 2. 활성화
conda activate infomae

# 3. PyTorch 설치 (CUDA)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 또는 CPU만
conda install pytorch torchvision cpuonly -c pytorch

# 4. 나머지 패키지 설치
pip install timm einops wandb opencv-python tensorboard pysaliency

# 5. 설치 확인
python test_installation.py
```

---

## 🚀 설치 후 빠른 테스트

### 1. 설치 검증 (필수)

```bash
python test_installation.py
```

모든 테스트가 통과해야 합니다:
- ✅ Directory Structure
- ✅ Basic Imports
- ✅ PyTorch & CUDA
- ✅ InfoMAE Modules
- ✅ Model Instantiation
- ✅ Data Loading

### 2. 빠른 실험 (CIFAR-100)

```bash
bash scripts/quick_start.sh
```

이 명령은:
- CIFAR-100 자동 다운로드
- Stage 2로 10 에포크 학습
- 약 20-30분 소요 (GPU 기준)

### 3. Pretrained 가중치 다운로드 (선택)

```bash
bash scripts/download_pretrained.sh
```

---

## 🐛 문제 해결

### Q1: "python3: command not found"

**해결:**
```bash
# macOS
brew install python3

# Ubuntu
sudo apt update
sudo apt install python3 python3-pip

# Windows
# Python 공식 사이트에서 설치: https://www.python.org/downloads/
```

### Q2: "conda: command not found"

**해결:**
```bash
# Miniconda 설치 (가볍고 빠름)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh

# 설치 후
source ~/.bashrc  # 또는 ~/.zshrc
```

### Q3: CUDA 관련 오류

**증상:**
```
RuntimeError: CUDA out of memory
```

**해결:**
```bash
# 1. Batch size 줄이기
python main.py --batch_size 128  # default: 256

# 2. GPU 메모리 확인
nvidia-smi

# 3. 다른 프로세스 종료
# 4. CPU로 실행
python main.py --device cpu
```

### Q4: "No module named 'xxx'"

**해결:**
```bash
# 가상환경이 활성화되었는지 확인
which python
# 출력: /path/to/venv/bin/python (또는 conda 경로)

# 패키지 재설치
pip install -r requirements.txt

# 특정 패키지만
pip install xxx
```

### Q5: Import 오류

**해결:**
```bash
# 올바른 디렉토리에 있는지 확인
pwd
# 출력: /Users/ihagmyeong/Documents/VClab/2DInfoMAE

cd /Users/ihagmyeong/Documents/VClab/2DInfoMAE

# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Q6: Permission denied

**해결:**
```bash
# 실행 권한 부여
chmod +x setup_venv.sh
chmod +x setup_conda.sh
chmod +x scripts/*.sh

# 또는 bash로 직접 실행
bash setup_venv.sh
```

### Q7: M1/M2 Mac에서 설치 문제

**해결:**
```bash
# Apple Silicon에 최적화된 PyTorch 설치
pip install torch torchvision

# 또는 conda
conda install pytorch torchvision -c pytorch

# OpenCV 문제시
pip install opencv-python-headless
```

---

## 🎓 추가 설정 (선택사항)

### Jupyter Notebook 지원

```bash
# Jupyter 설치
pip install jupyter ipykernel

# Kernel 등록
python -m ipykernel install --user --name=infomae --display-name="InfoMAE"

# Jupyter 실행
jupyter notebook
```

### Wandb 설정

```bash
# Wandb 로그인
wandb login

# API 키 입력 (https://wandb.ai/authorize 에서 확인)

# .env 파일 수정
nano .env
# WANDB_API_KEY=your_api_key
# WANDB_ENTITY=your_username
```

### GPU 다중 사용

```bash
# 특정 GPU 선택
export CUDA_VISIBLE_DEVICES=0,1

# .env 파일에 추가
echo "CUDA_VISIBLE_DEVICES=0,1" >> .env
```

---

## ✅ 최종 체크리스트

설치 완료 후 확인:

- [ ] Python 3.8+ 설치됨
- [ ] 가상환경 생성 및 활성화됨
- [ ] PyTorch 설치됨
- [ ] CUDA 작동 확인 (GPU 사용시)
- [ ] 모든 의존성 설치됨
- [ ] `test_installation.py` 통과
- [ ] 필요한 디렉토리 생성됨 (data, outputs, pretrained)
- [ ] 실행 스크립트 권한 부여됨

---

## 📚 다음 단계

환경 설정이 완료되었다면:

1. **README.md** 읽기 - 프로젝트 개요
2. **USAGE.md** 읽기 - 상세 사용법
3. **IMPLEMENTATION_CHECK.md** 읽기 - 구현 검증
4. **빠른 테스트 실행**:
   ```bash
   bash scripts/quick_start.sh
   ```
5. **전체 실험 실행**:
   ```bash
   bash run_experiments.sh
   ```

---

## 🆘 도움이 필요하신가요?

문제가 해결되지 않으면:

1. **에러 메시지 전체 복사**
2. **실행한 명령어 기록**
3. **환경 정보 확인**:
   ```bash
   python --version
   pip list
   nvidia-smi  # GPU 사용시
   ```
4. **GitHub Issues에 질문 올리기**

---

**Happy coding! 🚀**

