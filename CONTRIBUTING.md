# Contributing to Tri-Neuro Hybrid Architecture

Thank you for your interest in contributing to the Tri-Neuro Hybrid Architecture project! We welcome contributions from the research community.

## 👋 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- Git fundamentals
- Understanding of neural network architectures (helpful but not required)

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/sunghunkwag/tri-neuro-hybrid-architecture.git
cd tri-neuro-hybrid-architecture

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black isort flake8
```

## 🔧 How to Contribute

### Reporting Issues

- **Bug Reports**: Include system info, error messages, and reproduction steps
- **Feature Requests**: Describe the use case and expected behavior
- **Questions**: Use GitHub Discussions for general questions

### Code Contributions

1. **Fork the Repository**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add docstrings to all functions/classes

3. **Test Your Changes**
   ```bash
   pytest tests/
   python core_architecture.py  # Run basic smoke test
   ```

4. **Format Code**
   ```bash
   black .
   isort .
   flake8 .
   ```

5. **Submit Pull Request**
   - Clear title describing the change
   - Reference related issues
   - Include test results

## 🎯 Priority Areas

We're especially interested in contributions in these areas:

### High Priority

- [ ] **Transformer Adapter**: Full GPT/BERT integration
- [ ] **JEPA Implementation**: Integrate Meta's JEPA model
- [ ] **Liquid NN Integration**: Connect to Liquid AI's LFM
- [ ] **Benchmarking**: Standard AGI task evaluations
- [ ] **Documentation**: Tutorials and API references

### Medium Priority

- [ ] Multi-GPU training support
- [ ] Visualization tools for manifold states
- [ ] Example applications (robotics, NLP, vision)
- [ ] Performance optimization
- [ ] Unit test coverage

### Research Ideas

- [ ] Alternative routing mechanisms
- [ ] Self-supervised manifold learning
- [ ] Hierarchical module composition
- [ ] Meta-learning for router adaptation

## 📝 Code Style

### Python Standards

- **PEP 8** compliance
- **Type hints** for function signatures
- **Docstrings** in Google style format

### Example

```python
def project_to_manifold(
    self,
    source_type: str,
    data: torch.Tensor
) -> torch.Tensor:
    """
    Project modality-specific data to shared manifold.
    
    Args:
        source_type: One of ['semantic', 'spatial', 'dynamic']
        data: Input tensor from respective module
        
    Returns:
        Manifold representation tensor of shape [batch, manifold_dim]
        
    Raises:
        ValueError: If source_type is unknown
    """
    # Implementation...
```

## ✅ Testing Guidelines

### Writing Tests

- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test names

### Example Test

```python
import pytest
import torch
from core_architecture import TriNeuroSystem

def test_cognitive_cycle():
    """Test basic cognitive cycle execution."""
    system = TriNeuroSystem()
    inputs = {
        'text_embedding': torch.randn(1, 768),
        'visual_embedding': torch.randn(1, 1024)
    }
    state = system.cycle(inputs)
    
    assert state.shape == (1, 512)
    assert not torch.isnan(state).any()
```

## 💬 Communication

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Request Comments**: Implementation discussions

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🚀 Recognition

All contributors will be:
- Listed in project acknowledgments
- Mentioned in release notes (for significant contributions)
- Eligible for co-authorship on related publications (major contributions)

## ❓ Questions?

Not sure where to start? Open an issue with the `question` label or reach out through GitHub Discussions.

---

**Thank you for helping advance AGI research!** 🧠✨
