"""
Test package for SymQNet Molecular Optimization CLI

Constraints for testing:
- Model only supports up to 10 qubits
- L parameter must be 82 (64 + 18 metadata)
- Use CPU device for testing to avoid CUDA requirements
"""

# Test constants matching model constraints
MAX_QUBITS = 10
L_BASE = 64
METADATA_DIM = 18
L_TOTAL = L_BASE + METADATA_DIM  # 82

# Test configuration
TEST_DEVICE = 'cpu'
TEST_SHOTS = 100  # Low shots for fast tests
TEST_ROLLOUTS = 2  # Minimal rollouts for testing
