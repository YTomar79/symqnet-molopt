#!/usr/bin/env python3
"""
Add custom Hamiltonian to SymQNet-MolOpt

This command validates user Hamiltonian files and adds them to the system
for use with the optimization CLI.
"""

import click
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constraints
SUPPORTED_QUBITS = 10
USER_DIR = Path("user_hamiltonians")
EXAMPLES_DIR = Path("examples")

class HamiltonianValidator:
    """Validates user Hamiltonian files for compatibility"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate a Hamiltonian file for compatibility.
        
        Returns:
            (is_valid, data) tuple
        """
        self.errors = []
        self.warnings = []
        
        try:
            # Load and parse JSON
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Required field validation
            if not self._validate_required_fields(data):
                return False, {}
            
            # Qubit count validation (CRITICAL)
            if not self._validate_qubit_count(data):
                return False, {}
            
            # Pauli terms validation
            if not self._validate_pauli_terms(data):
                return False, {}
            
            # Optional field validation (warnings only)
            self._validate_optional_fields(data)
            
            # Structure validation
            self._validate_hamiltonian_structure(data)
            
            return True, data
            
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON format: {e}")
            return False, {}
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")
            return False, {}
    
    def _validate_required_fields(self, data: Dict[str, Any]) -> bool:
        """Validate required JSON fields"""
        required_fields = ['format', 'molecule', 'n_qubits', 'pauli_terms']
        
        for field in required_fields:
            if field not in data:
                self.errors.append(f"Missing required field: '{field}'")
                return False
        
        return True
    
    def _validate_qubit_count(self, data: Dict[str, Any]) -> bool:
        """Validate qubit count is exactly 10"""
        n_qubits = data.get('n_qubits', 0)
        
        if not isinstance(n_qubits, int):
            self.errors.append(f"'n_qubits' must be an integer, got {type(n_qubits).__name__}")
            return False
        
        if n_qubits != SUPPORTED_QUBITS:
            self.errors.append(
                f"❌ INCOMPATIBLE: SymQNet-MolOpt only supports {SUPPORTED_QUBITS}-qubit systems.\n"
                f"   Your Hamiltonian: {n_qubits} qubits\n"
                f"   Required: {SUPPORTED_QUBITS} qubits\n\n"
                f"💡 To fix this:\n"
                f"   • Map your system to {SUPPORTED_QUBITS} qubits using active space approximation\n"
                f"   • Use Jordan-Wigner encoding with appropriate truncation\n"
                f"   • Apply symmetry reduction techniques"
            )
            return False
        
        return True
    
    def _validate_pauli_terms(self, data: Dict[str, Any]) -> bool:
        """Validate Pauli terms structure and content"""
        pauli_terms = data.get('pauli_terms', [])
        
        if not isinstance(pauli_terms, list):
            self.errors.append("'pauli_terms' must be a list")
            return False
        
        if len(pauli_terms) == 0:
            self.errors.append("'pauli_terms' cannot be empty")
            return False
        
        n_qubits = data['n_qubits']
        
        for i, term in enumerate(pauli_terms):
            if not isinstance(term, dict):
                self.errors.append(f"Pauli term {i} must be a dictionary")
                return False
            
            # Check required fields in each term
            if 'coefficient' not in term:
                self.errors.append(f"Pauli term {i} missing 'coefficient'")
                return False
            
            if 'pauli_string' not in term:
                self.errors.append(f"Pauli term {i} missing 'pauli_string'")
                return False
            
            # Validate coefficient
            coeff = term['coefficient']
            if not isinstance(coeff, (int, float, complex)):
                self.errors.append(f"Pauli term {i}: coefficient must be a number")
                return False
            
            # Validate Pauli string
            pauli_str = term['pauli_string']
            if not isinstance(pauli_str, str):
                self.errors.append(f"Pauli term {i}: 'pauli_string' must be a string")
                return False
            
            if len(pauli_str) != n_qubits:
                self.errors.append(
                    f"Pauli term {i}: string length {len(pauli_str)} != {n_qubits} qubits"
                )
                return False
            
            # Check valid Pauli characters
            valid_chars = set('IXYZ')
            for j, char in enumerate(pauli_str.upper()):
                if char not in valid_chars:
                    self.errors.append(
                        f"Pauli term {i}, position {j}: invalid character '{char}'. "
                        f"Must be one of: {valid_chars}"
                    )
                    return False
        
        return True
    
    def _validate_optional_fields(self, data: Dict[str, Any]):
        """Validate optional fields and issue warnings"""
        
        # Check for description
        if 'description' not in data:
            self.warnings.append("Consider adding a 'description' field for documentation")
        
        # Check for true parameters (useful for validation)
        if 'true_parameters' not in data:
            self.warnings.append(
                "Consider adding 'true_parameters' for validation and benchmarking"
            )
        
        # Check format field value
        format_val = data.get('format', '')
        if format_val not in ['openfermion', 'qiskit', 'custom']:
            self.warnings.append(
                f"Unusual format '{format_val}'. "
                f"Consider using: 'openfermion', 'qiskit', or 'custom'"
            )
    
    def _validate_hamiltonian_structure(self, data: Dict[str, Any]):
        """Analyze and validate Hamiltonian structure"""
        pauli_terms = data['pauli_terms']
        
        # Count interaction types
        single_qubit_terms = 0
        two_qubit_terms = 0
        multi_qubit_terms = 0
        
        for term in pauli_terms:
            pauli_str = term['pauli_string'].upper()
            non_identity = sum(1 for char in pauli_str if char != 'I')
            
            if non_identity == 1:
                single_qubit_terms += 1
            elif non_identity == 2:
                two_qubit_terms += 1
            elif non_identity > 2:
                multi_qubit_terms += 1
        
        # Structure warnings
        if single_qubit_terms == 0:
            self.warnings.append("No single-qubit terms found - unusual for molecular systems")
        
        if two_qubit_terms == 0:
            self.warnings.append("No two-qubit interactions found - check if this is intended")
        
        if multi_qubit_terms > len(pauli_terms) * 0.5:
            self.warnings.append(
                f"Many multi-qubit terms ({multi_qubit_terms}) - "
                f"may increase computational complexity"
            )
        
        # Check for identity term
        has_identity = any(
            term['pauli_string'].upper() == 'I' * SUPPORTED_QUBITS
            for term in pauli_terms
        )
        
        if not has_identity:
            self.warnings.append(
                "No identity term found - consider adding constant energy offset"
            )


def generate_safe_filename(original_name: str, data: Dict[str, Any]) -> str:
    """Generate safe filename for user Hamiltonian"""
    
    # Clean the original name
    molecule = data.get('molecule', 'unknown')
    safe_molecule = "".join(c for c in molecule if c.isalnum() or c in "-_")
    
    # Add hash for uniqueness
    content_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:8]
    
    # Generate filename
    filename = f"{safe_molecule}_{SUPPORTED_QUBITS}q_{content_hash}.json"
    
    return filename


@click.command()
@click.argument('hamiltonian_file', type=click.Path(exists=True, path_type=Path))
@click.option('--output-name', '-n',
              type=str,
              help='Custom name for the added file (without extension)')
@click.option('--to-examples', '-e',
              is_flag=True,
              help='Add to examples directory instead of user directory')
@click.option('--force', '-f',
              is_flag=True,
              help='Overwrite existing file if it exists')
@click.option('--validate-only', '-v',
              is_flag=True,
              help='Only validate, do not add to system')
@click.option('--quiet', '-q',
              is_flag=True,
              help='Minimal output')
def main(hamiltonian_file: Path, output_name: str, to_examples: bool, 
         force: bool, validate_only: bool, quiet: bool):
    """
    Add custom Hamiltonian file to SymQNet-MolOpt system.
    
    Validates the Hamiltonian file and adds it to the appropriate directory
    for use with the optimization CLI.
    
    Example:
        symqnet-add my_molecule.json
        symqnet-add custom_system.json --output-name my_system
        symqnet-add research.json --to-examples
    """
    
    if not quiet:
        print("🔍 SymQNet-MolOpt Hamiltonian Validator")
        print("=" * 50)
    
    # Validate the file
    validator = HamiltonianValidator()
    is_valid, data = validator.validate_file(hamiltonian_file)
    
    # Print validation results
    if not quiet:
        print(f"📁 File: {hamiltonian_file}")
        print(f"📏 Size: {hamiltonian_file.stat().st_size / 1024:.1f} KB")
    
    if validator.errors:
        print("\n❌ VALIDATION FAILED:")
        for error in validator.errors:
            print(f"   {error}")
        
        if not quiet:
            print(f"\n💡 Fix these issues and try again.")
        
        raise click.ClickException("Validation failed")
    
    if validator.warnings and not quiet:
        print("\n⚠️  WARNINGS:")
        for warning in validator.warnings:
            print(f"   {warning}")
    
    if not quiet:
        print(f"\n✅ VALIDATION PASSED:")
        print(f"   Molecule: {data.get('molecule', 'unknown')}")
        print(f"   Qubits: {data['n_qubits']}")
        print(f"   Pauli terms: {len(data['pauli_terms'])}")
        print(f"   Format: {data.get('format', 'unknown')}")
    
    # If validation-only, stop here
    if validate_only:
        if not quiet:
            print(f"\n🎯 File is compatible with SymQNet-MolOpt!")
        return
    
    # Determine target directory and filename
    if to_examples:
        target_dir = EXAMPLES_DIR
        if not quiet:
            print(f"\n📁 Adding to examples directory...")
    else:
        target_dir = USER_DIR
        target_dir.mkdir(exist_ok=True)
        if not quiet:
            print(f"\n📁 Adding to user directory...")
    
    # Generate filename
    if output_name:
        target_filename = f"{output_name}.json"
    else:
        target_filename = generate_safe_filename(hamiltonian_file.name, data)
    
    target_path = target_dir / target_filename
    
    # Check if file exists
    if target_path.exists() and not force:
        print(f"\n⚠️  File already exists: {target_path}")
        if click.confirm("Overwrite?"):
            force = True
        else:
            raise click.ClickException("File exists. Use --force to overwrite.")
    
    # Copy file
    try:
        shutil.copy2(hamiltonian_file, target_path)
        
        if not quiet:
            print(f"✅ Successfully added: {target_path}")
            
            # Show usage instructions
            print(f"\n🚀 USAGE:")
            print(f"   symqnet-molopt --hamiltonian {target_path} --output results.json")
            
            print(f"\n📋 QUICK COMMANDS:")
            print(f"   # Quick test:")
            print(f"   symqnet-molopt --hamiltonian {target_path} --shots 256 --output quick_test.json")
            print(f"   ")
            print(f"   # Research quality:")
            print(f"   symqnet-molopt --hamiltonian {target_path} --shots 2048 --output research.json --n-rollouts 15")
            
        else:
            print(f"✅ Added: {target_path}")
            
    except Exception as e:
        raise click.ClickException(f"Failed to copy file: {e}")


if __name__ == '__main__':
    main()
