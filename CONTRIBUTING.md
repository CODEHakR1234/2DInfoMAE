# Contributing to InfoMAE

Thank you for your interest in contributing to InfoMAE! 🎉

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code contributions.

---

## 🐛 Reporting Bugs

If you find a bug, please open an issue with:

1. **Clear title**: Describe the bug concisely
2. **Description**: What happened vs. what you expected
3. **Steps to reproduce**: Minimal code to reproduce the issue
4. **Environment**:
   ```bash
   python --version
   pip list | grep torch
   nvidia-smi  # if using GPU
   ```
5. **Error messages**: Full traceback if applicable

---

## 💡 Feature Requests

We're open to new ideas! Please open an issue describing:

1. **Use case**: What problem does it solve?
2. **Proposed solution**: How would you implement it?
3. **Alternatives**: Other approaches you considered
4. **Additional context**: Examples, papers, etc.

---

## 🔧 Contributing Code

### Setting Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/InfoMAE.git
cd InfoMAE

# Create a new branch
git checkout -b feature/your-feature-name

# Set up environment
bash setup_venv.sh
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install black flake8 pytest  # code quality tools

# Run tests
python test_installation.py
```

### Code Style

We follow PEP 8 style guidelines:

```bash
# Format code
black *.py models/*.py data/*.py utils/*.py

# Check style
flake8 --max-line-length=120 *.py
```

### Making Changes

1. **Write tests**: Add tests for new features
2. **Document**: Update docstrings and README if needed
3. **Test locally**: Run `python test_installation.py`
4. **Commit**: Write clear commit messages
   ```bash
   git commit -m "feat: add new surprisal computation method"
   git commit -m "fix: resolve CUDA memory issue in adaptive masking"
   git commit -m "docs: update installation guide for M1 Macs"
   ```

### Commit Message Format

We use conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### Pull Request Process

1. **Update documentation**: If you changed functionality
2. **Add tests**: Ensure all tests pass
3. **Update CHANGELOG**: Add entry for your changes
4. **Create PR**: 
   - Clear title describing the change
   - Reference related issues
   - Describe what and why
   - Include before/after if relevant

5. **Review process**:
   - Maintainers will review your PR
   - Address feedback
   - Once approved, we'll merge!

---

## 📚 Documentation

Improvements to documentation are always welcome:

- Fix typos
- Clarify confusing sections
- Add examples
- Translate to other languages

---

## 🧪 Running Tests

```bash
# Installation test
python test_installation.py

# Quick functional test
bash scripts/quick_start.sh

# Full test suite (if you add pytest)
pytest tests/
```

---

## 📋 Areas for Contribution

Here are some areas where we'd especially appreciate help:

### High Priority
- [ ] Distributed training (Multi-GPU/DDP)
- [ ] Mixed precision training (AMP)
- [ ] Complete saliency evaluation pipeline
- [ ] Additional datasets support
- [ ] Performance optimizations

### Medium Priority
- [ ] Jupyter notebook tutorials
- [ ] Colab demo
- [ ] Docker container
- [ ] Pre-commit hooks
- [ ] CI/CD pipeline

### Documentation
- [ ] Video tutorials
- [ ] More visualization examples
- [ ] Troubleshooting guide
- [ ] Translation to other languages

### Research Extensions
- [ ] Different attention mechanisms
- [ ] Alternative masking strategies
- [ ] Multi-modal extensions
- [ ] Comparison with other methods

---

## 🤝 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all:

- **Be respectful**: Value each other's ideas, styles, and viewpoints
- **Be considerate**: Your work will be used by others, and you depend on others' work
- **Be collaborative**: Collaboration reduces redundancy and improves quality
- **Be open**: We welcome all skill levels and backgrounds

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information
- Other unethical or unprofessional conduct

---

## ❓ Questions?

Feel free to:

- Open an issue with the `question` label
- Join discussions in existing issues
- Reach out to maintainers

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Thank You!

Every contribution helps make InfoMAE better. We appreciate your time and effort!

**Happy coding! 🚀**

