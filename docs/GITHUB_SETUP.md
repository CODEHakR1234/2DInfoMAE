# 🚀 GitHub 업로드 가이드

InfoMAE 프로젝트를 GitHub에 올리는 단계별 가이드입니다.

---

## ✅ 준비 완료 체크리스트

다음 항목들이 이미 준비되었습니다:

- [x] `.gitignore` 설정 완료
- [x] `.gitattributes` 설정 완료
- [x] Git 저장소 초기화 완료
- [x] GitHub 이슈/PR 템플릿 추가
- [x] CONTRIBUTING.md 추가
- [x] CHANGELOG.md 추가
- [x] GitHub Actions CI/CD 설정
- [x] 불필요한 파일 제거 (PROJECT_SUMMARY.txt)
- [x] 모든 문서 정리 완료

---

## 📝 Step 1: GitHub에서 새 Repository 생성

1. [GitHub](https://github.com)에 로그인
2. 우측 상단 '+' 클릭 → 'New repository' 선택
3. Repository 설정:
   - **Repository name**: `InfoMAE` (또는 원하는 이름)
   - **Description**: `Information-Driven Masked Autoencoding for Human-Like Visual Attention`
   - **Public** 또는 **Private** 선택
   - **❌ Initialize with README 체크 해제** (이미 로컬에 있음)
   - **❌ Add .gitignore 체크 해제** (이미 있음)
   - **License**: MIT (선택사항, 이미 로컬에 있음)
4. 'Create repository' 클릭

---

## 🔧 Step 2: Git 초기 커밋

터미널에서 실행:

```bash
cd /Users/ihagmyeong/Documents/VClab/2DInfoMAE

# Git 사용자 설정 (처음 한 번만)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 모든 파일 추가
git add .

# 초기 커밋
git commit -m "feat: initial commit - InfoMAE v1.0.0

- Implement Surprisal-Weighted Attention (SWA)
- Implement Adaptive Masking
- Implement Information Bottleneck regularization
- Add 4-stage progressive training (Stage 0-3)
- Support ImageNet-100, CIFAR-100, STL-10
- Add comprehensive documentation (8 docs)
- Add automated setup scripts (venv/conda)
- Add evaluation and visualization tools
- Add GitHub templates and CI/CD"

# 현재 상태 확인
git status
```

---

## 🌐 Step 3: GitHub에 푸시

GitHub에서 생성한 repository의 URL을 사용:

```bash
# Remote 추가 (YOUR-USERNAME을 실제 GitHub 사용자명으로 변경)
git remote add origin https://github.com/YOUR-USERNAME/InfoMAE.git

# 또는 SSH 사용시:
# git remote add origin git@github.com:YOUR-USERNAME/InfoMAE.git

# Remote 확인
git remote -v

# Main 브랜치로 변경 (GitHub 기본 브랜치)
git branch -M main

# 푸시
git push -u origin main
```

**성공!** 🎉 이제 GitHub에서 프로젝트를 확인할 수 있습니다.

---

## 🏷️ Step 4: Release 생성 (선택사항)

GitHub에서 첫 번째 릴리스를 만들어보세요:

1. Repository 페이지에서 'Releases' 클릭
2. 'Create a new release' 클릭
3. 릴리스 설정:
   - **Tag version**: `v1.0.0`
   - **Release title**: `InfoMAE v1.0.0 - Initial Release`
   - **Description**: CHANGELOG.md의 v1.0.0 섹션 복사
   - **Set as the latest release** 체크
4. 'Publish release' 클릭

---

## 📋 Step 5: Repository 설정

### 5.1 Topics 추가

Repository 페이지에서 'About' 옆 톱니바퀴 클릭:

**추천 Topics:**
```
deep-learning
computer-vision
pytorch
masked-autoencoder
self-supervised-learning
attention-mechanism
vision-transformer
representation-learning
information-theory
visual-attention
```

### 5.2 Repository 설명 추가

**Description:**
```
Information-Driven Masked Autoencoding for Human-Like Visual Attention
```

**Website:**
```
https://github.com/YOUR-USERNAME/InfoMAE
```

### 5.3 Features 활성화

Repository Settings에서:
- [x] Issues (이슈 추적)
- [x] Projects (프로젝트 관리)
- [x] Wiki (문서)
- [x] Discussions (토론)

---

## 🔐 Step 6: Branch Protection (선택사항)

Main 브랜치를 보호하려면:

1. Settings → Branches
2. 'Add rule'
3. 'Branch name pattern': `main`
4. 설정:
   - [x] Require a pull request before merging
   - [x] Require status checks to pass before merging
   - [x] Require conversation resolution before merging
5. 'Create' 클릭

---

## 📊 Step 7: GitHub Actions 확인

1. Repository에서 'Actions' 탭 클릭
2. 'I understand my workflows, go ahead and enable them' 클릭
3. 자동으로 테스트가 실행됩니다

Workflow는 다음 경우에 실행됩니다:
- `main` 또는 `develop` 브랜치에 push
- `main` 또는 `develop`로 Pull Request

---

## 🎨 Step 8: README 배지 추가 (선택사항)

README.md 상단에 배지를 추가할 수 있습니다:

```markdown
# InfoMAE

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/YOUR-USERNAME/InfoMAE/workflows/Tests/badge.svg)](https://github.com/YOUR-USERNAME/InfoMAE/actions)
[![GitHub Stars](https://img.shields.io/github/stars/YOUR-USERNAME/InfoMAE.svg)](https://github.com/YOUR-USERNAME/InfoMAE/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/YOUR-USERNAME/InfoMAE.svg)](https://github.com/YOUR-USERNAME/InfoMAE/network)
```

---

## 🌟 Step 9: Repository 홍보

프로젝트를 공유하세요:

### Academic Communities
- [ ] [Papers with Code](https://paperswithcode.com/) - 논문 제출시
- [ ] [Reddit r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
- [ ] Twitter (X) - #DeepLearning #ComputerVision 태그

### ML Communities
- [ ] [Hugging Face](https://huggingface.co/) - 모델 공유
- [ ] [PyTorch Forums](https://discuss.pytorch.org/)
- [ ] [ML Discord Servers](https://discord.com/)

---

## 🔄 Step 10: 향후 업데이트 워크플로우

### 코드 변경시:

```bash
# 변경사항 확인
git status

# 변경된 파일 추가
git add .

# 커밋 (conventional commits 사용)
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in adaptive masking"
git commit -m "docs: update installation guide"

# 푸시
git push origin main
```

### 새 기능 개발시 (브랜치 사용):

```bash
# 새 브랜치 생성
git checkout -b feature/new-feature

# 작업 후 커밋
git add .
git commit -m "feat: implement new feature"

# 푸시
git push origin feature/new-feature

# GitHub에서 Pull Request 생성
```

---

## 📚 Step 11: Git LFS 설정 (대용량 파일용)

모델 가중치 같은 큰 파일을 올리려면:

```bash
# Git LFS 설치 (한 번만)
git lfs install

# 큰 파일 추적
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.h5"

# .gitattributes 커밋
git add .gitattributes
git commit -m "chore: add Git LFS tracking"
git push

# 이제 큰 파일 추가 가능
git add pretrained/model.pth
git commit -m "add: pretrained model weights"
git push
```

**주의:** GitHub LFS는 무료 한도가 있습니다 (1GB storage, 1GB bandwidth/month)

---

## ⚠️ 주의사항

### ❌ 절대 커밋하면 안 되는 것:

- API 키 / 비밀번호
- `.env` 파일 (이미 .gitignore에 있음)
- 개인 데이터
- 대용량 데이터셋 (링크만 제공)
- 임시 파일 / 로그 파일

### ✅ 커밋해야 하는 것:

- 소스 코드 (`.py` 파일)
- 문서 (`.md` 파일)
- 설정 파일
- 스크립트 (`.sh` 파일)
- Requirements

---

## 🆘 문제 해결

### Q: 푸시가 거부되었습니다

```bash
# 해결: Remote의 변경사항 먼저 가져오기
git pull origin main --rebase
git push origin main
```

### Q: 실수로 큰 파일을 커밋했습니다

```bash
# 최근 커밋 취소 (푸시 전)
git reset HEAD~1

# 파일 제거
git rm --cached large_file.pth

# .gitignore에 추가
echo "large_file.pth" >> .gitignore

# 다시 커밋
git add .
git commit -m "fix: remove large file"
```

### Q: 민감한 정보를 실수로 커밋했습니다

```bash
# BFG Repo-Cleaner 사용 (권장)
# https://rtyley.github.io/bfg-repo-cleaner/

# 또는 git filter-branch (복잡함)
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive-file" \
  --prune-empty --tag-name-filter cat -- --all
```

**중요:** 민감한 정보가 노출되었다면 즉시 키를 재발급하세요!

---

## ✅ 최종 체크리스트

GitHub 업로드 전:

- [ ] 모든 민감한 정보 제거 확인
- [ ] README.md 작성 완료
- [ ] .gitignore 설정 확인
- [ ] 라이센스 파일 포함
- [ ] 테스트 통과 확인
- [ ] 문서 최종 검토

GitHub 업로드 후:

- [ ] Repository 생성 완료
- [ ] 첫 커밋 푸시 완료
- [ ] Topics 추가 완료
- [ ] Description 설정 완료
- [ ] Actions 활성화 완료
- [ ] (선택) Release 생성 완료
- [ ] (선택) Branch protection 설정

---

## 🎉 완료!

축하합니다! InfoMAE 프로젝트가 GitHub에 성공적으로 업로드되었습니다.

**다음 단계:**
1. 프로젝트 홍보
2. 이슈/PR 관리
3. 커뮤니티 구축
4. 지속적인 개선

**Repository URL:**
```
https://github.com/YOUR-USERNAME/InfoMAE
```

Happy coding! 🚀

