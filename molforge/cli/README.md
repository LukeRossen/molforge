# MolForge CLI

Command-line interface for MolForge - Modular Molecular Data Processing Pipeline.

## Installation

Install MolForge with CLI support:

```bash
pip install -e ".[cli]"
```

This installs the `molforge` command with optional PyYAML support for configuration files.

## Quick Start

```bash
# Show help
molforge --help

# Display architecture diagram
molforge --diagram

# Run pipeline with ChEMBL target
molforge run CHEMBL234

# Generate configuration template
molforge init --template basic --output config.yaml

# Run with configuration file
molforge run CHEMBL234 --config config.yaml

# Batch processing
molforge run CHEMBL234 CHEMBL279 CHEMBL280 --batch
```

## Commands

### `molforge run`

Execute MolForge pipeline on input data.

**Usage:**
```bash
molforge run <input> [options]
```

**Arguments:**
- `input`: ChEMBL ID(s), CSV file(s), or other data source

**Options:**
- `--config, -c`: Configuration file (YAML)
- `--output, -o`: Output directory
- `--steps`: Comma-separated list of steps (e.g., "source,curate,confs")
- `--batch, -b`: Process multiple inputs in batch mode
- `--quiet, -q`: Suppress progress messages

**Examples:**
```bash
# Single ChEMBL target
molforge run CHEMBL234

# From CSV file
molforge run compounds.csv --output ./results

# With configuration
molforge run CHEMBL234 --config pipeline.yaml

# Batch processing
molforge run CHEMBL234 CHEMBL279 --batch
```

### `molforge init`

Generate configuration template.

**Usage:**
```bash
molforge init [options]
```

**Options:**
- `--template, -t`: Template type (basic, full, conformers)
- `--output, -o`: Output file path (default: molforge_config.yaml)

**Examples:**
```bash
# Basic template
molforge init --template basic

# Full pipeline template
molforge init --template full --output my_config.yaml

# Conformer generation template
molforge init --template conformers
```

**Templates:**
- **basic**: Source → ChEMBL → Curate (minimal pipeline)
- **full**: All pipeline steps (source, chembl, curate, tokens, distributions, confs)
- **conformers**: Conformer generation pipeline with RDKit backend

### `molforge info`

Display system information.

**Usage:**
```bash
molforge info [target]
```

**Targets:**
- `version`: Show MolForge version (default)
- `actors`: List available actors
- `backends`: Show backend options
- `examples`: Display usage examples

**Options:**
- `--actor`: Filter backends by actor name (with `backends` target)

**Examples:**
```bash
# Show version
molforge info
molforge info version

# List actors
molforge info actors

# Show backends
molforge info backends

# Backends for specific actor
molforge info backends --actor confs

# Usage examples
molforge info examples
```

### `molforge validate`

Validate configuration file.

**Usage:**
```bash
molforge validate <config>
```

**Arguments:**
- `config`: Configuration file to validate

**Options:**
- `--quiet, -q`: Suppress detailed output

**Examples:**
```bash
# Validate configuration
molforge validate config.yaml

# Quiet validation
molforge validate config.yaml --quiet
```

## Configuration Files

Configuration files use YAML format to define pipeline parameters.

### Basic Configuration

```yaml
# Pipeline steps
steps:
  - source
  - chembl
  - curate

# Actor configurations
source:
  backend: sql
  db_path: chembl_36.db

chembl:
  activity_types: [IC50, Ki]
  min_pchembl: 5.0

curate:
  mol_steps: [desalt, neutralize, sanitize]
  stereo_policy: assign

# Output settings
output:
  dir: ./molforge_output
  format: csv
  checkpoints: false
```

### Full Configuration

See `molforge init --template full` for complete example with all actors.

## Global Options

Available for all commands:

- `--version`: Show version and exit
- `--diagram`: Display architecture diagram and exit
- `--help, -h`: Show help message

## Display Features

The CLI provides:

- **ASCII Art Banner**: MolForge logo on startup
- **Colored Output**: Success (green), errors (red), warnings (yellow), info (cyan)
- **Progress Indicators**: Step-by-step pipeline execution tracking
- **Architecture Diagram**: Visual representation of pipeline flow

## Testing

Run CLI tests:

```bash
# All CLI tests
pytest tests/cli/ -v

# Specific test module
pytest tests/cli/test_display.py -v
pytest tests/cli/test_commands.py -v
pytest tests/cli/test_integration.py -v
```

## Python API

The CLI complements the Python API. Both interfaces are available:

```python
# Python API
from molforge import MolForge, PipeParams

params = PipeParams(steps=['source', 'curate'])
forge = MolForge(params)
df = forge.forge('CHEMBL234')
```

```bash
# CLI equivalent
molforge run CHEMBL234 --steps source,curate
```

## Dependencies

- **Core**: argparse (built-in)
- **Optional**: pyyaml (for config file support)

Install with CLI support:
```bash
pip install -e ".[cli]"
```

## Architecture

```
molforge/cli/
├── __init__.py       # Package initialization
├── main.py           # Entry point & argument parsing
├── commands.py       # Command implementations
├── display.py        # Display utilities & formatting
└── templates/        # Configuration templates
    ├── basic.yaml
    ├── conformers.yaml
    └── full.yaml
```

## Development

The CLI module is designed to be non-invasive:
- No modifications to existing MolForge code
- Optional installation (`pip install -e ".[cli]"`)
- Standalone test suite in `tests/cli/`

## Troubleshooting

**Command not found:**
```bash
# Reinstall with CLI support
pip install -e ".[cli]"
```

**YAML errors:**
```bash
# Install PyYAML
pip install pyyaml
```

**Import errors:**
```bash
# Ensure molforge is installed
pip install -e .
```

## See Also

- [Main README](../README.md): MolForge overview
- [Architecture Diagram](../ARCHITECTURE_FLOW_DIAGRAM.md): Technical documentation
- [Refactor Status](../MOLFORGE_REFACTOR_STATUS.md): Development status
