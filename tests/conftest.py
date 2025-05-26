"""
Pytest configuration for SymQNet tests
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def device():
    """Provide CPU device for testing"""
    return torch.device('cpu')


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_hamiltonian():
    """Provide mock Hamiltonian for testing"""
    return {
        "format": "openfermion",
        "molecule": "test",
        "n_qubits": 4,
        "pauli_terms": [
            {"coefficient": -1.0, "pauli_string": "IIII"},
            {"coefficient": 0.5, "pauli_string": "ZIII"},
            {"coefficient": 0.3, "pauli_string": "ZZII"}
        ],
        "true_parameters": {
            "coupling": [0.3],
            "field": [0.5, 0.0, 0.0, 0.0]
        }
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests"""
    torch.manual_seed(42)
    np.random.seed(42)


def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add slow marker to tests that might take time
    for item in items:
        if "integration" in item.nodeid or "symqnet" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
