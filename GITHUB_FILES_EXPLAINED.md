# 📚 GitHub 특수 파일 설명

GitHub에 올릴 때 추가한 특수 파일들이 무엇인지 쉽게 설명합니다.

---

## 📁 `.github/` 폴더란?

**간단히**: GitHub에서 자동으로 인식하는 "설정 폴더"입니다.

이 폴더 안에 있는 파일들은 GitHub가 특별하게 취급합니다:
- 이슈 템플릿
- PR(Pull Request) 템플릿
- 자동화 워크플로우
- 등등...

**비유**: 프로젝트의 "GitHub 전용 설정 폴더"

---

## 🗂️ `.github/` 폴더 구조

```
.github/
├── ISSUE_TEMPLATE/           # 이슈 템플릿 모음
│   ├── bug_report.md        # 버그 리포트 양식
│   └── feature_request.md   # 기능 요청 양식
│
├── PULL_REQUEST_TEMPLATE.md # PR 작성 양식
│
├── workflows/               # GitHub Actions (자동화)
│   └── test.yml            # 자동 테스트 설정
│
└── FUNDING.yml             # 후원 버튼 설정
```

---

## 🐛 1. `ISSUE_TEMPLATE/bug_report.md`

### 무엇인가요?
누군가 버그를 발견했을 때 **쉽게 리포트하도록 돕는 양식**입니다.

### 실제 동작:
GitHub에서 'New Issue' 클릭 → **자동으로 이 양식이 나타남**

**예시:**
```markdown
## 🐛 버그 설명
[여기에 버그 설명]

## 재현 방법
1. 이 명령 실행: `python main.py`
2. 이 에러 발생: `...`

## 환경
- OS: Ubuntu 22.04
- Python: 3.9
- GPU: RTX 4090
```

### 왜 필요한가?
❌ **없으면**: "안 돼요" 같은 애매한 이슈만 올라옴
✅ **있으면**: 체계적인 버그 리포트 → 빠른 해결

---

## ✨ 2. `ISSUE_TEMPLATE/feature_request.md`

### 무엇인가요?
새로운 기능 제안을 받을 때 쓰는 **표준 양식**

### 실제 동작:
'New Issue' → 'Feature Request' 선택 → 자동으로 양식 제공

**예시:**
```markdown
## 💡 제안하는 기능
Distributed training 지원

## 왜 필요한가요?
Multi-GPU로 학습하고 싶습니다

## 제안하는 방법
PyTorch DDP 사용...
```

### 장점:
- 제안이 구조화됨
- 중복 제안 방지
- 논의가 체계적

---

## 📝 3. `PULL_REQUEST_TEMPLATE.md`

### 무엇인가요?
코드를 수정해서 올릴 때(Pull Request) **자동으로 나타나는 체크리스트**

### 실제 동작:
누군가 PR 생성 → **자동으로 이 템플릿이 채워짐**

**예시:**
```markdown
## 📝 설명
버그 수정: adaptive masking 메모리 누수 해결

## 타입
- [x] 🐛 버그 수정
- [ ] ✨ 새 기능

## 테스트
- [x] test_installation.py 통과
- [x] quick_start.sh 실행 확인

## 체크리스트
- [x] 코드 스타일 확인
- [x] 문서 업데이트
- [x] 테스트 추가
```

### 왜 필요한가?
코드 리뷰할 때 **무엇을 확인했는지 명확**하게 보여줌

---

## 🤖 4. `workflows/test.yml` (GitHub Actions)

### 무엇인가요?
**자동 테스트 시스템** - 코드가 올라올 때마다 자동으로 테스트 실행!

### 실제 동작:

```
[코드를 GitHub에 push]
        ↓
[GitHub Actions 자동 실행]
        ↓
[여러 환경에서 테스트]
  • Ubuntu + Python 3.8
  • Ubuntu + Python 3.9
  • macOS + Python 3.9
        ↓
[결과를 자동으로 표시]
  ✅ 모든 테스트 통과!
  또는
  ❌ Python 3.8에서 실패!
```

### 실제 파일 내용:

```yaml
name: Tests

on:
  push:                    # 코드 push할 때마다
    branches: [ main ]
  pull_request:            # PR 올릴 때마다
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest # Ubuntu에서 실행
    steps:
      - name: 코드 가져오기
        uses: actions/checkout@v3
      
      - name: Python 설치
        uses: actions/setup-python@v4
        
      - name: 의존성 설치
        run: pip install -r requirements.txt
        
      - name: 테스트 실행
        run: python test_installation.py
```

### 장점:
- **자동**: 수동으로 테스트 안 해도 됨
- **빠름**: 문제를 즉시 발견
- **신뢰**: 여러 환경에서 동작 확인

### GitHub에서 보는 방법:
Repository → 'Actions' 탭 → 테스트 결과 확인

---

## 💰 5. `FUNDING.yml`

### 무엇인가요?
**후원 버튼** 설정 파일

### 실제 동작:
Repository에 "Sponsor" 버튼이 나타남

```yaml
# GitHub Sponsors
github: [your-username]

# Other platforms
patreon: your-patreon
ko_fi: your-kofi
```

### 결과:
<img src="sponsor-button.png" /> ← 이런 버튼이 repository에 생김

### 필요성:
선택사항! 후원 받고 싶으면 설정, 아니면 비워두면 됨

---

## 📄 `.gitattributes`란?

### 무엇인가요?
Git이 **파일을 어떻게 다룰지** 알려주는 설정 파일

**비유**: 프로젝트의 "파일 취급 설명서"

### 주요 기능:

#### 1️⃣ **줄바꿈 통일** (가장 중요!)

```gitattributes
# 모든 텍스트 파일을 LF(Unix)로 통일
*.py text eol=lf
*.sh text eol=lf
*.md text eol=lf
```

**문제 상황:**
- Windows: 줄바꿈 = `\r\n` (CRLF)
- Mac/Linux: 줄바꿈 = `\n` (LF)

**해결:**
`.gitattributes`로 통일! → 모든 OS에서 같은 코드

#### 2️⃣ **바이너리 파일 인식**

```gitattributes
# 이미지는 바이너리로 취급
*.png binary
*.jpg binary

# 모델 파일도 바이너리
*.pth binary
*.pt binary
```

**왜?**: Git이 바이너리를 텍스트로 처리하면 깨짐 방지

#### 3️⃣ **대용량 파일 처리 (Git LFS)**

```gitattributes
# 큰 파일은 Git LFS 사용
*.pth filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
```

**설명:**
- 보통 Git: 100MB 이상 업로드 불가
- Git LFS: 큰 파일도 OK

#### 4️⃣ **언어별 Diff 설정**

```gitattributes
*.py diff=python
```

코드 변경사항을 볼 때 Python 문법에 맞게 보여줌

---

## 🤔 실전 예시로 이해하기

### 예시 1: 버그 발견했을 때

**없으면:**
```
Issue: "안 돼요"
```
→ 개발자 당황 😵

**있으면:**
```
[자동으로 템플릿 제공]

버그: CUDA out of memory
재현: python main.py --batch_size 256
환경: RTX 3090, CUDA 11.8
에러 로그: [전체 로그]
```
→ 개발자가 바로 이해하고 해결 🎯

---

### 예시 2: 코드 수정해서 올릴 때

**없으면:**
```
PR: "버그 고쳤어요"
```
→ 뭘 테스트했는지 모름 🤷

**있으면:**
```
[자동 템플릿]

수정 내용: adaptive masking 버그 수정
테스트:
  ✅ 설치 테스트 통과
  ✅ 빠른 테스트 실행
  ✅ 문서 업데이트
```
→ 리뷰어가 안심하고 병합 ✅

---

### 예시 3: GitHub Actions

**상황:** 실수로 버그 있는 코드를 push

**자동 동작:**
```
1. [Push 감지]
2. [자동 테스트 시작]
3. [테스트 실패 발견]
4. [이메일/알림 발송]
   ❌ test_installation.py 실패!
   import torch 에러
```

→ 즉시 발견하고 수정 가능! 🚨

---

## 📊 정리 표

| 파일 | 역할 | 필수? | 효과 |
|------|------|-------|------|
| `ISSUE_TEMPLATE/bug_report.md` | 버그 리포트 양식 | 선택 | 체계적인 버그 추적 |
| `ISSUE_TEMPLATE/feature_request.md` | 기능 제안 양식 | 선택 | 구조화된 제안 |
| `PULL_REQUEST_TEMPLATE.md` | PR 체크리스트 | 선택 | 코드 리뷰 품질 향상 |
| `workflows/test.yml` | 자동 테스트 | 선택 | CI/CD 자동화 |
| `FUNDING.yml` | 후원 버튼 | 선택 | 후원 받기 |
| `.gitattributes` | 파일 처리 규칙 | 권장 | 크로스 플랫폼 호환성 |

---

## 💡 실용적인 조언

### 처음 GitHub 사용자라면:

**꼭 필요한 것:**
- `.gitignore` ⭐⭐⭐ (필수!)
- `README.md` ⭐⭐⭐ (필수!)

**있으면 좋은 것:**
- `.gitattributes` ⭐⭐ (권장)
- `LICENSE` ⭐⭐ (권장)

**나중에 추가해도 되는 것:**
- `.github/` 폴더 전체 ⭐ (선택)
- GitHub Actions ⭐ (선택)

### 우리 프로젝트의 경우:

모든 파일을 추가했지만, **당장 사용하지 않아도 괜찮습니다!**

**실제로 동작하는 시점:**
1. **이슈 템플릿**: 누군가 'New Issue' 클릭할 때
2. **PR 템플릿**: Pull Request 만들 때
3. **GitHub Actions**: 코드를 push할 때
4. **FUNDING**: 설정 완료 후 Sponsor 버튼 표시

**지금 당장은:**
- 그냥 GitHub에 올려도 OK!
- 나중에 필요하면 활용하면 됨

---

## 🎯 요약

### `.github/` 폴더:
```
GitHub 전용 설정들을 모아둔 폴더
→ 이슈/PR 템플릿, 자동화 등
→ 있으면 프로페셔널하지만 없어도 작동함
```

### `.gitattributes`:
```
Git이 파일을 다루는 방법 설정
→ 줄바꿈 통일, 바이너리 인식 등
→ 있으면 크로스 플랫폼 문제 예방
```

### 결론:
**모두 "편의 기능"입니다!**
- 있으면 협업이 편하고 전문적으로 보임
- 없어도 GitHub 업로드는 가능
- 우리는 이미 다 만들어뒀으니 그냥 올리면 됨 👍

---

## ❓ 자주 묻는 질문

**Q: 이 파일들 없이 올려도 되나요?**
A: 네! 핵심 코드와 README만 있어도 OK입니다.

**Q: 나중에 추가할 수 있나요?**
A: 네! 언제든지 추가/수정 가능합니다.

**Q: 이거 복잡한데 다 이해해야 하나요?**
A: 아뇨! 처음엔 몰라도 됩니다. 필요할 때 배우면 됩니다.

**Q: 지금 당장 뭐 해야 하나요?**
A: 그냥 `GITHUB_SETUP.md` 따라서 업로드하면 끝!

---

**간단 요약: "GitHub를 더 잘 쓰기 위한 보너스 파일들"** ✨

