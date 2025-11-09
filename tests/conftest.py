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
    """Provide mock 10-qubit Hamiltonian for testing"""
    # 🔧 FIX: Use 10 qubits to match SymQNet constraint
    return {
        "format": "openfermion",
        "molecule": "test_10q",
        "n_qubits": 10,  # must be 10 for SymQNet
        "pauli_terms": [
            {"coefficient": -2.0, "pauli_string": "IIIIIIIIII"},
            {"coefficient": 0.5, "pauli_string": "ZIIIIIIIII"},
            {"coefficient": 0.3, "pauli_string": "IZIIIIIIIII"},
            {"coefficient": 0.2, "pauli_string": "ZZIIIIIIIII"},
            {"coefficient": 0.15, "pauli_string": "XIIIIIIIIX"},
            {"coefficient": 0.15, "pauli_string": "YIIIIIIIYY"}
        ],
        "true_parameters": {
            "coupling": [0.2, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 9 coupling params
            "field": [0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # 10 field params
        },
        "description": "Mock 10-qubit Hamiltonian for testing SymQNet"
    }


@pytest.fixture
def mock_hamiltonian_small():
    """Provide small invalid Hamiltonian for testing validation errors"""
    # This should FAIL validation - useful for testing error handling
    return {
        "format": "openfermion",
        "molecule": "invalid_test",
        "n_qubits": 4,  # Intentionally wrong for testing validation
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


@pytest.fixture
def mock_model_paths(temp_dir):
    """Provide mock model file paths for testing"""
    models_dir = temp_dir / "models"
    models_dir.mkdir()
    
    # Create dummy model files
    vae_path = models_dir / "vae_M10_f.pth"
    symqnet_path = models_dir / "FINAL_FIXED_SYMQNET.pth"
    
    # Create minimal dummy state dicts
    dummy_vae_state = {
        'enc_fc1.weight': torch.randn(128, 10),
        'enc_fc1.bias': torch.zeros(128),
        'enc_fc2.weight': torch.randn(128, 128),
        'enc_fc2.bias': torch.zeros(128),
        'enc_mu.weight': torch.randn(64, 128),
        'enc_mu.bias': torch.zeros(64),
        'enc_logsigma.weight': torch.randn(64, 128),
        'enc_logsigma.bias': torch.zeros(64),
        'dec_fc1.weight': torch.randn(128, 64),
        'dec_fc1.bias': torch.zeros(128),
        'dec_fc2.weight': torch.randn(128, 128),
        'dec_fc2.bias': torch.zeros(128),
        'dec_out.weight': torch.randn(10, 128),
        'dec_out.bias': torch.zeros(10),
    }
    
    dummy_symqnet_state = {
        'model_state_dict': {
            'dummy_param': torch.tensor(1.0)
        }
    }
    
    torch.save(dummy_vae_state, vae_path)
    torch.save(dummy_symqnet_state, symqnet_path)
    
    return {
        'vae_path': vae_path,
        'symqnet_path': symqnet_path,
        'models_dir': models_dir
    }


@pytest.fixture
def mock_examples_dir(temp_dir):
    """Create mock examples directory with 10-qubit files"""
    examples_dir = temp_dir / "examples"
    examples_dir.mkdir()
    
    # Create valid 10-qubit example
    valid_example = {
        "format": "custom",
        "molecule": "H2O_test",
        "n_qubits": 10,
        "pauli_terms": [
            {"coefficient": -75.0, "pauli_string": "IIIIIIIIII"},
            {"coefficient": 0.3, "pauli_string": "IIIIIIIIIZ"},
            {"coefficient": 0.2, "pauli_string": "IIIIIIIIZZ"}
        ],
        "true_parameters": {
            "coupling": [0.2] + [0.0] * 8,
            "field": [0.3] + [0.0] * 9
        }
    }
    
    valid_file = examples_dir / "H2O_10q.json"
    with open(valid_file, 'w') as f:
        import json
        json.dump(valid_example, f, indent=2)
    
    return {
        'examples_dir': examples_dir,
        'valid_file': valid_file
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests"""
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def symqnet_config():
    """Provide SymQNet configuration parameters"""
    return {
        'n_qubits': 10,
        'L': 64,  # Base latent dimension
        'T': 10,
        'M_evo': 5,
        'A': 150,  # 10 * 3 * 5
        'meta_dim': 18,  # 10 + 3 + 5
        'total_params': 19  # 9 coupling + 10 field
    }


def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "model_loading: marks tests that require model files"
    )
    config.addinivalue_line(
        "markers", "validation: marks tests for input validation"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add slow marker to tests that might take time
    for item in items:
        if "integration" in item.nodeid or "symqnet" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add model_loading marker to tests that load models
        if "model" in item.nodeid.lower() and "load" in item.nodeid.lower():
            item.add_marker(pytest.mark.model_loading)
        
        # Add validation marker to validation tests
        if "valid" in item.nodeid.lower():
            item.add_marker(pytest.mark.validation)


# Custom assertion helpers
def assert_valid_10qubit_hamiltonian(data):
    """Assert that Hamiltonian data is valid for 10-qubit system"""
    assert data['n_qubits'] == 10, f"Expected 10 qubits, got {data['n_qubits']}"
    assert 'pauli_terms' in data, "Missing pauli_terms"
    assert len(data['pauli_terms']) > 0, "Empty pauli_terms"
    
    for i, term in enumerate(data['pauli_terms']):
        assert 'coefficient' in term, f"Term {i} missing coefficient"
        assert 'pauli_string' in term, f"Term {i} missing pauli_string"
        assert len(term['pauli_string']) == 10, f"Term {i} wrong string length"


def assert_valid_parameter_estimates(estimates, expected_params=19):
    """Assert that parameter estimates have correct structure"""
    assert isinstance(estimates, (list, np.ndarray)), "Estimates must be array-like"
    if isinstance(estimates, list):
        estimates = np.array(estimates)
    
    assert len(estimates) == expected_params, f"Expected {expected_params} parameters, got {len(estimates)}"
    assert np.all(np.isfinite(estimates)), "All estimates must be finite"


# Test data generators
def generate_test_measurement_data(n_samples=100):
    """Generate test measurement data for VAE training"""
    np.random.seed(42)
    data = []
    
    for _ in range(n_samples):
        # Generate realistic measurement expectation values
        measurements = np.random.uniform(-1, 1, 10)  # 10 measurements
        data.append(measurements.astype(np.float32))
    
    return data


# Environment variables for testing
os.environ['SYMQNET_TEST_MODE'] = '1'  # Flag for test mode
os.environ['SYMQNET_DEVICE'] = 'cpu'    # Force CPU for tests
