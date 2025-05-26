#!/usr/bin/env python3
"""
Complete system validation for SymQNet CLI

This script performs comprehensive validation of the entire CLI installation,
including file structure, dependencies, model loading, and functionality.
"""

import subprocess
import json
import os
import sys
import importlib
from pathlib import Path
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🔍 {title.upper()}")
    print(f"{'='*60}")

def print_result(test_name, passed, details=""):
    """Print test result with formatting"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{test_name:<40} | {status}")
    if details and not passed:
        print(f"    └─ {details}")

def run_command(cmd, description):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timeout"
    except Exception as e:
        return False, str(e)

def check_file_exists(filepath, description):
    """Check if file exists and return status"""
    exists = Path(filepath).exists()
    size = Path(filepath).stat().st_size if exists else 0
    details = f"Size: {size/1024/1024:.1f}MB" if size > 1024*1024 else f"Size: {size/1024:.1f}KB"
    return exists, details if exists else "File not found"

def validate_file_structure():
    """Validate project file structure"""
    print_header("File Structure Validation")
    
    required_files = [
        ("architectures.py", "Your exact neural network architectures"),
        ("cli.py", "Main CLI entry point"),
        ("hamiltonian_parser.py", "Molecular Hamiltonian parser"),
        ("measurement_simulator.py", "Quantum measurement simulator"),
        ("policy_engine.py", "SymQNet policy integration"),
        ("bootstrap_estimator.py", "Uncertainty quantification"),
        ("utils.py", "Utility functions"),
        ("requirements.txt", "Python dependencies"),
        ("models/vae_M10_f.pth", "Pre-trained VAE model"),
        ("models/FINAL_FIXED_SYMQNET.pth", "Trained SymQNet model"),
        # 🔧 FIX: Check for 10-qubit examples only
        ("examples/H2O_10q.json", "H2O molecule example (10 qubits)"),
        ("scripts/create_examples.py", "Example generator script"),
        ("scripts/test_models.py", "Model testing script")
    ]
    
    all_exist = True
    for filepath, description in required_files:
        exists, details = check_file_exists(filepath, description)
        print_result(f"{filepath:<30}", exists, details)
        all_exist = all_exist and exists
    
    return all_exist

def validate_python_environment():
    """Validate Python environment and dependencies"""
    print_header("Python Environment Validation")
    
    # Check Python version
    py_version = sys.version_info
    py_ok = py_version >= (3, 8)
    print_result("Python version (>=3.8)", py_ok, f"Found: {py_version.major}.{py_version.minor}")
    
    # Check required packages
    required_packages = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("click", "CLI framework"),
        ("scipy", "Scientific computing"),
        ("matplotlib", "Plotting"),
        ("pandas", "Data analysis"),
        ("tqdm", "Progress bars"),
        ("gym", "RL environment")
    ]
    
    all_imports_ok = True
    for package, description in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print_result(f"{package:<20}", True, f"v{version}")
        except ImportError as e:
            print_result(f"{package:<20}", False, f"Import error: {e}")
            all_imports_ok = False
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    cuda_details = f"Devices: {torch.cuda.device_count()}" if cuda_available else "CPU only"
    print_result("CUDA support", cuda_available, cuda_details)
    
    return py_ok and all_imports_ok

def validate_architecture_imports():
    """Validate architecture imports from your exact code"""
    print_header("Architecture Import Validation")
    
    architectures_to_test = [
        "VariationalAutoencoder",
        "GraphEmbed", 
        "TemporalContextualAggregator",
        "PolicyValueHead",
        "FixedSymQNetWithEstimator",
        "SpinChainEnv",
        "get_pauli_matrices"
    ]
    
    all_imports_ok = True
    
    # 🔧 FIX: Safer import testing
    try:
        import architectures
        print_result("architectures module", True, "Module imported successfully")
        
        for arch_name in architectures_to_test:
            try:
                arch_obj = getattr(architectures, arch_name)
                print_result(f"{arch_name:<30}", True, "Available in module")
            except AttributeError:
                print_result(f"{arch_name:<30}", False, f"Not found in architectures module")
                all_imports_ok = False
                
    except ImportError as e:
        print_result("architectures module", False, f"Import error: {e}")
        all_imports_ok = False
        
        # Try individual imports as fallback
        for arch_name in architectures_to_test:
            print_result(f"{arch_name:<30}", False, "Module import failed")
    
    return all_imports_ok

def validate_model_loading():
    """Validate model loading with exact architectures"""
    print_header("Model Loading Validation")
    
    device = torch.device('cpu')  # Use CPU for validation
    
    # Test VAE loading
    vae_ok = False
    try:
        from architectures import VariationalAutoencoder
        vae = VariationalAutoencoder(M=10, L=64).to(device)
        vae.load_state_dict(torch.load('models/vae_M10_f.pth', map_location=device))
        vae.eval()
        
        # Test forward pass
        test_input = torch.randn(10)
        with torch.no_grad():
            mu, logvar = vae.encode(test_input)
            z = vae.reparameterize(mu, logvar)
        
        vae_ok = True
        print_result("VAE model loading", True, f"Output shape: {z.shape}")
    except Exception as e:
        print_result("VAE model loading", False, str(e))
    
    # Test SymQNet loading
    symqnet_ok = False
    try:
        from architectures import FixedSymQNetWithEstimator
        
        checkpoint = torch.load('models/FINAL_FIXED_SYMQNET.pth', map_location=device)
        
        # Model parameters from your exact training
        n_qubits = 10
        L = 64  # 🔧 FIX: Correct base dimension
        T = 10
        M_evo = 5
        A = n_qubits * 3 * M_evo
        
        # Graph connectivity from your training
        edges = [(i, i+1) for i in range(n_qubits-1)] + [(i+1, i) for i in range(n_qubits-1)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32, device=device) * 0.1
        
        symqnet = FixedSymQNetWithEstimator(
            vae=vae,
            n_qubits=n_qubits,
            L=L,  # 🔧 FIX: Use L=64, not L=82
            edge_index=edge_index,
            edge_attr=edge_attr,
            T=T,
            A=A,
            M_evo=M_evo,
            K_gnn=2
        ).to(device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            symqnet.load_state_dict(checkpoint['model_state_dict'])
        else:
            symqnet.load_state_dict(checkpoint)
        
        symqnet.eval()
        
        # Test forward pass
        obs = torch.randn(10)
        metadata = torch.zeros(18)  # n_qubits + 3 + M_evo
        
        with torch.no_grad():
            dist, value, theta_hat = symqnet(obs, metadata)
        
        symqnet_ok = True
        print_result("SymQNet model loading", True, f"Params: {theta_hat.shape}")
        
    except Exception as e:
        print_result("SymQNet model loading", False, str(e))
    
    return vae_ok and symqnet_ok

def validate_cli_functionality():
    """Validate CLI functionality"""
    print_header("CLI Functionality Validation")
    
    # Test CLI help
    help_ok, help_error = run_command("python cli.py --help", "CLI help")
    print_result("CLI help command", help_ok, help_error if not help_ok else "")
    
    # Test component imports
    imports = [
        "from hamiltonian_parser import HamiltonianParser",
        "from measurement_simulator import MeasurementSimulator", 
        "from policy_engine import PolicyEngine",
        "from bootstrap_estimator import BootstrapEstimator",
        "from utils import setup_logging"
    ]
    
    all_imports_ok = True
    for imp in imports:
        try:
            exec(imp)
            module_name = imp.split()[-1]
            print_result(f"Import {module_name}", True)
        except Exception as e:
            print_result(f"Import {module_name}", False, str(e))
            all_imports_ok = False
    
    return help_ok and all_imports_ok

def validate_examples():
    """Validate example Hamiltonian files"""
    print_header("Example Files Validation")
    
    # 🔧 FIX: Only check for 10-qubit examples
    example_files = [
        "examples/H2O_10q.json"
    ]
    
    all_examples_ok = True
    
    for example_file in example_files:
        if not Path(example_file).exists():
            print_result(f"{example_file}", False, "File not found")
            all_examples_ok = False
            continue
        
        try:
            with open(example_file, 'r') as f:
                data = json.load(f)
            
            # Validate structure
            required_keys = ['format', 'molecule', 'n_qubits', 'pauli_terms']
            has_required = all(key in data for key in required_keys)
            
            if has_required:
                n_qubits = data['n_qubits']
                n_terms = len(data['pauli_terms'])
                
                # 🔧 FIX: Validate 10-qubit constraint
                if n_qubits == 10:
                    print_result(f"{example_file}", True, f"{n_qubits}q, {n_terms} terms")
                else:
                    print_result(f"{example_file}", False, f"Wrong qubit count: {n_qubits} != 10")
                    all_examples_ok = False
            else:
                print_result(f"{example_file}", False, "Missing required keys")
                all_examples_ok = False
        
        except Exception as e:
            print_result(f"{example_file}", False, str(e))
            all_examples_ok = False
    
    # Check if we can create examples if none exist
    if not all_examples_ok:
        print_result("Creating examples", True, "Run 'symqnet-examples' to create")
    
    return all_examples_ok

def run_integration_test():
    """Run a minimal integration test"""
    print_header("Integration Test")
    
    try:
        # 🔧 FIX: Create 10-qubit example first if needed
        h2o_example = Path("examples/H2O_10q.json")
        if not h2o_example.exists():
            print_result("Creating test example", True, "No 10-qubit example found")
            # Skip integration test if no valid example
            print_result("Integration test", False, "No valid 10-qubit examples available")
            return False
        
        # Create a minimal test with 10-qubit example
        cmd = """python cli.py \
            --hamiltonian examples/H2O_10q.json \
            --shots 50 \
            --output outputs/validation_test.json \
            --max-steps 3 \
            --n-rollouts 1 \
            --device cpu"""
        
        # Create outputs directory
        Path("outputs").mkdir(exist_ok=True)
        
        print("Running minimal CLI test (this may take a moment)...")
        success, error = run_command(cmd, "Minimal CLI test")
        
        if success:
            # Check if output file was created
            output_exists = Path("outputs/validation_test.json").exists()
            if output_exists:
                with open("outputs/validation_test.json", 'r') as f:
                    result = json.load(f)
                
                has_results = 'symqnet_results' in result
                print_result("Integration test", has_results, "Complete pipeline works")
                return has_results
            else:
                print_result("Integration test", False, "No output file created")
                return False
        else:
            print_result("Integration test", False, error)
            return False
    
    except Exception as e:
        print_result("Integration test", False, str(e))
        return False

def main():
    """Run complete validation suite"""
    print("🔬 SYMQNET CLI VALIDATION SUITE")
    print(f"Validation started at: {Path.cwd()}")
    
    # Run all validation tests
    tests = [
        ("File Structure", validate_file_structure),
        ("Python Environment", validate_python_environment),
        ("Architecture Imports", validate_architecture_imports),
        ("Model Loading", validate_model_loading),
        ("CLI Functionality", validate_cli_functionality),
        ("Example Files", validate_examples),
        ("Integration Test", run_integration_test)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_result(test_name, False, f"Test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print_header("Validation Summary")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} | {status}")
    
    print(f"\n📊 OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED! Your SymQNet CLI is ready to use.")
        print("\n💡 Quick start:")
        print("   python cli.py --hamiltonian examples/H2O_10q.json --shots 512 --output test.json")
    else:
        print("⚠️  Some validations failed. Please fix the issues above before using the CLI.")
        
        # Provide specific guidance
        if not results.get("File Structure", True):
            print("\n🔧 File Structure Issues:")
            print("   • Ensure all required files are in the correct locations")
            print("   • Check that model files exist in models/ directory")
            print("   • Run 'symqnet-examples' to create 10-qubit examples")
        
        if not results.get("Model Loading", True):
            print("\n🔧 Model Loading Issues:")
            print("   • Verify your trained models are compatible")
            print("   • Check that architectures.py contains your exact code")
        
        if not results.get("Integration Test", True):
            print("\n🔧 Integration Issues:")
            print("   • Run: python scripts/test_models.py")
            print("   • Check device compatibility (try --device cpu)")
            print("   • Ensure 10-qubit examples exist")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
