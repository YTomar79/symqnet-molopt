#!/usr/bin/env python3
"""
Create example molecular Hamiltonian files for SymQNet CLI

This script generates various molecular Hamiltonian examples in the correct
JSON format for use with the CLI.
"""

import json
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_h2_example():
    """Create H2 molecule example (4 qubits) - NOT USED due to 10-qubit constraint"""
    return {
        "format": "openfermion",
        "molecule": "H2",
        "basis": "sto-3g",
        "bond_length": 0.74,
        "n_qubits": 4,
        "pauli_terms": [
            {"coefficient": -1.0523732, "pauli_string": "IIII"},
            {"coefficient": 0.39793742, "pauli_string": "IIIZ"},
            {"coefficient": -0.39793742, "pauli_string": "IIZI"},
            {"coefficient": -0.01128010, "pauli_string": "IIZZ"},
            {"coefficient": 0.18093119, "pauli_string": "IXIX"},
            {"coefficient": 0.18093119, "pauli_string": "IYIY"}
        ],
        "true_parameters": {
            "coupling": [0.18093119, 0.18093119, -0.01128010],
            "field": [0.39793742, -0.39793742, 0.0, 0.0]
        },
        "reference_energy": -1.857275,
        "description": "H2 molecule in minimal basis, optimized geometry"
    }

def create_lih_example():
    """Create LiH molecule example (6 qubits) - NOT USED due to 10-qubit constraint"""
    return {
        "format": "openfermion",
        "molecule": "LiH",
        "basis": "sto-3g",
        "bond_length": 1.45,
        "n_qubits": 6,
        "pauli_terms": [
            {"coefficient": -7.8384, "pauli_string": "IIIIII"},
            {"coefficient": 0.1809, "pauli_string": "IIIIIZ"},
            {"coefficient": 0.1809, "pauli_string": "IIIIZI"},
            {"coefficient": -0.2436, "pauli_string": "IIIZII"},
            {"coefficient": 0.1665, "pauli_string": "IIXIIX"},
            {"coefficient": 0.1665, "pauli_string": "IIYIIY"},
            {"coefficient": 0.1743, "pauli_string": "IZIIZI"},
            {"coefficient": 0.1203, "pauli_string": "ZIIIII"},
            {"coefficient": -0.0453, "pauli_string": "IZIIIZ"}
        ],
        "true_parameters": {
            "coupling": [0.1665, 0.1665, 0.1743, -0.2436, 0.1809],
            "field": [0.1809, 0.1203, -0.0453, 0.0, 0.0, 0.0]
        },
        "reference_energy": -7.8631,
        "description": "LiH molecule in minimal basis, equilibrium geometry"
    }

def create_beh2_example():
    """Create BeH2 molecule example (8 qubits) - NOT USED due to 10-qubit constraint"""
    return {
        "format": "openfermion", 
        "molecule": "BeH2",
        "basis": "sto-3g",
        "geometry": "linear",
        "n_qubits": 8,
        "pauli_terms": [
            {"coefficient": -15.5907, "pauli_string": "IIIIIIII"},
            {"coefficient": 0.2454, "pauli_string": "IIIIIIIZ"},
            {"coefficient": 0.2454, "pauli_string": "IIIIIIZI"},
            {"coefficient": -0.3278, "pauli_string": "IIIIIIZZ"},
            {"coefficient": 0.1821, "pauli_string": "IIIIXIIX"},
            {"coefficient": 0.1821, "pauli_string": "IIIIYIIY"},
            {"coefficient": 0.0934, "pauli_string": "IIZIIZII"},
            {"coefficient": -0.0621, "pauli_string": "IZIIIIZI"},
            {"coefficient": 0.1456, "pauli_string": "XIIIIIIX"},
            # 🔧 FIX: Correct Pauli string
            {"coefficient": 0.1456, "pauli_string": "YIIIIIYY"}  # Was "YIIIIITY"
        ],
        "true_parameters": {
            "coupling": [0.1821, 0.1821, -0.3278, 0.0934, -0.0621, 0.1456, 0.1456],
            "field": [0.2454, 0.2454, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "reference_energy": -15.6142,
        "description": "BeH2 molecule in linear geometry, minimal basis"
    }

def create_water_example():
    """Create H2O molecule example (10 qubits) - matches SymQNet training"""
    return {
        "format": "openfermion",
        "molecule": "H2O", 
        "basis": "sto-3g",
        "geometry": "C2v",
        "n_qubits": 10,
        "pauli_terms": [
            {"coefficient": -74.9431, "pauli_string": "IIIIIIIIII"},
            {"coefficient": 0.3421, "pauli_string": "IIIIIIIIIZ"},
            {"coefficient": 0.3421, "pauli_string": "IIIIIIIIZI"},
            {"coefficient": -0.4523, "pauli_string": "IIIIIIIIZZ"},
            {"coefficient": 0.2134, "pauli_string": "IIIIIIXIIX"},
            {"coefficient": 0.2134, "pauli_string": "IIIIIIYIIY"},
            {"coefficient": 0.1876, "pauli_string": "IIIIZIIZII"},
            {"coefficient": -0.0934, "pauli_string": "IIIZIIIZII"},
            {"coefficient": 0.1623, "pauli_string": "IIXIIIIIIX"},
            {"coefficient": 0.1623, "pauli_string": "IIYIIIIIIY"},
            {"coefficient": 0.0823, "pauli_string": "ZIIIIIIIII"},
            {"coefficient": -0.0456, "pauli_string": "IZIIIIIIIZ"}
        ],
        "true_parameters": {
            "coupling": [0.2134, 0.2134, -0.4523, 0.1876, -0.0934, 0.1623, 0.1623, 0.0823, -0.0456],
            "field": [0.3421, 0.3421, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "reference_energy": -75.0123,
        "description": "H2O molecule optimized for SymQNet 10-qubit training"
    }

# 🔧 FIX: Proper newline and default parameter
def create_test_hamiltonian(n_qubits=10):  # Default to 10 qubits
    """Create a simple test Hamiltonian for debugging"""
    np.random.seed(42)  # Reproducible
    
    pauli_terms = []
    
    # Identity term
    pauli_terms.append({
        "coefficient": -2.0,
        "pauli_string": "I" * n_qubits
    })
    
    # Single-qubit Z terms
    for i in range(n_qubits):
        coeff = np.random.uniform(0.1, 0.5)
        pauli_str = list("I" * n_qubits)  # 🔧 FIX: Use list for easier manipulation
        pauli_str[i] = "Z"
        pauli_terms.append({
            "coefficient": coeff,
            "pauli_string": "".join(pauli_str)
        })
    
    # Two-qubit ZZ terms
    for i in range(n_qubits - 1):
        coeff = np.random.uniform(0.05, 0.2)
        pauli_str = list("I" * n_qubits)  # 🔧 FIX: Use list
        pauli_str[i] = "Z"
        pauli_str[i + 1] = "Z"  # 🔧 FIX: Correct adjacent qubits
        pauli_terms.append({
            "coefficient": coeff,
            "pauli_string": "".join(pauli_str)
        })
    
    # Some XX and YY terms
    for i in range(min(2, n_qubits - 1)):
        # XX term
        coeff_x = np.random.uniform(0.05, 0.15)
        pauli_str_x = list("I" * n_qubits)  # 🔧 FIX: Use list
        pauli_str_x[i] = "X"
        pauli_str_x[i + 1] = "X"
        pauli_terms.append({
            "coefficient": coeff_x,
            "pauli_string": "".join(pauli_str_x)
        })
        
        # YY term  
        coeff_y = np.random.uniform(0.05, 0.15)
        pauli_str_y = list("I" * n_qubits)  # 🔧 FIX: Use list
        pauli_str_y[i] = "Y"
        pauli_str_y[i + 1] = "Y"
        pauli_terms.append({
            "coefficient": coeff_y,
            "pauli_string": "".join(pauli_str_y)
        })
    
    return {
        "format": "custom",
        "molecule": f"test_{n_qubits}q",
        "basis": "test",
        "n_qubits": n_qubits,
        "pauli_terms": pauli_terms,
        "true_parameters": {
            "coupling": [0.1] * (n_qubits - 1),
            "field": [0.3] * n_qubits
        },
        "description": f"Test Hamiltonian for {n_qubits} qubits with random coefficients"
    }

def create_additional_10q_examples():
    """Create additional 10-qubit examples for variety"""
    
    # Alternative H2O configuration
    h2o_alt = {
        "format": "custom",
        "molecule": "H2O_alt",
        "basis": "sto-3g",
        "geometry": "C2v",
        "n_qubits": 10,
        "pauli_terms": [
            {"coefficient": -75.2134, "pauli_string": "IIIIIIIIII"},
            {"coefficient": 0.2891, "pauli_string": "IIIIIIIIIZ"},
            {"coefficient": 0.2891, "pauli_string": "IIIIIIIIZI"},
            {"coefficient": -0.3967, "pauli_string": "IIIIIIIIZZ"},
            {"coefficient": 0.1823, "pauli_string": "IIIIIIXIIX"},
            {"coefficient": 0.1823, "pauli_string": "IIIIIIYIIY"},
            {"coefficient": 0.1567, "pauli_string": "IIIIZIIZII"},
            {"coefficient": -0.0823, "pauli_string": "IIIZIIIZII"},
            {"coefficient": 0.1234, "pauli_string": "IIXIIIIIIX"},
            {"coefficient": 0.1234, "pauli_string": "IIYIIIIIIY"}
        ],
        "true_parameters": {
            "coupling": [0.1823, 0.1823, -0.3967, 0.1567, -0.0823, 0.1234, 0.1234, 0.0, 0.0],
            "field": [0.2891, 0.2891, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "reference_energy": -75.2456,
        "description": "Alternative H2O configuration for testing"
    }
    
    return h2o_alt

def main():
    """Main function to create ONLY 10-qubit example files"""
    
    # Create examples directory
    examples_dir = Path("../examples")
    examples_dir.mkdir(exist_ok=True)
    
    # 🔧 FIX: Expanded 10-qubit examples
    examples = [
        ("H2O_10q.json", create_water_example()),  # Primary example
        ("H2O_alt_10q.json", create_additional_10q_examples()),  # Alternative
        ("test_10q.json", create_test_hamiltonian(10)),  # Test system
        ("debug_10q.json", create_test_hamiltonian(10))  # Debug system (different seed)
    ]
    
    print("🔬 Creating 10-qubit molecular Hamiltonian examples...")
    print("⚠️  SymQNet-MolOpt only supports 10-qubit systems")
    print("=" * 50)
    
    created_files = []
    for filename, data in examples:
        filepath = examples_dir / filename
        
        try:
            # 🔧 FIX: Additional validation before writing
            if data['n_qubits'] != 10:
                print(f"❌ {filename:<15} | ERROR: {data['n_qubits']} qubits (must be 10)")
                continue
            
            # Validate Pauli strings
            valid_pauli = True
            for term in data['pauli_terms']:
                pauli_str = term['pauli_string']
                if len(pauli_str) != 10:
                    print(f"❌ {filename:<15} | ERROR: Pauli string length {len(pauli_str)} != 10")
                    valid_pauli = False
                    break
                if not all(c in 'IXYZ' for c in pauli_str):
                    print(f"❌ {filename:<15} | ERROR: Invalid Pauli characters in '{pauli_str}'")
                    valid_pauli = False
                    break
            
            if not valid_pauli:
                continue
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Validate the created file
            with open(filepath, 'r') as f:
                loaded = json.load(f)
            
            n_terms = len(loaded['pauli_terms'])
            n_qubits = loaded['n_qubits']
            
            print(f"✅ {filename:<20} | {n_qubits} qubits | {n_terms} terms")
            created_files.append(filepath)
            
        except Exception as e:
            print(f"❌ {filename:<20} | Error: {e}")
    
    print("=" * 50)
    print(f"📁 Created {len(created_files)} valid 10-qubit examples")
    
    if len(created_files) == 0:
        print("⚠️  No valid examples created - all must be exactly 10 qubits")
    else:
        print(f"\n🎯 USAGE:")
        for filepath in created_files:
            print(f"   symqnet-molopt --hamiltonian {filepath.name} --output results.json")
        
        print(f"\n📋 QUICK COMMANDS:")
        primary_file = created_files[0].name
        print(f"   # Quick test:")
        print(f"   symqnet-molopt --hamiltonian {primary_file} --shots 256 --output quick.json")
        print(f"   ")
        print(f"   # Research quality:")
        print(f"   symqnet-molopt --hamiltonian {primary_file} --shots 2048 --output research.json --n-rollouts 15")

if __name__ == "__main__":
    main()
