# SymQNet-MolOpt

Quantum neural network-enabled optimization for molecular Hamiltonian parameter estimation.

## Installation

pip install symqnet-molopt

text

## How to Use

### Validate Example Hamiltonians
symqnet-add examples/H2O_10q.json --validate-only

text

### Create Example Files
symqnet-examples

text

### Run the Optimization
symqnet-molopt
--hamiltonian examples/H2O_10q.json
--output results.json
--shots 1024
--n-rollouts 5
--max-steps 50

text

### Add Your Own Hamiltonian
symqnet-add my_molecule.json --validate-only

text
*(Only 10-qubit Hamiltonians are supported.)*

## Command Reference

| Command            | Description                                  |
|--------------------|----------------------------------------------|
| `symqnet-molopt`   | Run molecular parameter optimization         |
| `symqnet-add`      | Add or validate a Hamiltonian file           |
| `symqnet-examples` | Generate example Hamiltonians                |
| `symqnet-validate` | Validate installation                        |

## Key Optimization Parameters

| Parameter      | Description                |
|----------------|---------------------------|
| `--hamiltonian`| Input Hamiltonian file    |
| `--output`     | Output results file       |
| `--shots`      | Shots per measurement     |
| `--n-rollouts` | Number of optimization runs|
| `--max-steps`  | Max steps per rollout     |

## Quick Example

symqnet-molopt --hamiltonian examples/test_10q.json --output output.json --shots 512

text

## Requirements

- **Python 3.8+**
- **PyTorch 1.12+**
- **NumPy, SciPy, Click**
- **10-qubit Hamiltonians only**

## Support

- **Issues:** [GitHub Issues](https://github.com/YTomar79/symqnet-molopt/issues)
- **Help:** Run any command with `--help` for usage info.

## License

MIT License - See [LICENSE](LICENSE) file for details.
