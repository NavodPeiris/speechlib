# Code Quality Analysis Tools

## Overview

Automated tools for detecting redundant tests, dead code, and architectural violations.

## TestIQ - Test Redundancy Analysis

### Purpose
Detects tests that cover identical or overlapping code paths.

### Installation
```bash
pip install testiq
```

### Usage

Generate coverage:
```bash
pytest tests/ --testiq-output=testiq_coverage.json
```

Analyze redundancy:
```bash
# Windows (requires UTF-8 encoding)
python -X utf8 -c "from testiq.cli import main; main(['analyze', 'testiq_coverage.json'])"

# Linux/Mac
testiq analyze testiq_coverage.json
```

### Output Interpretation

| Metric | Meaning |
|--------|---------|
| Exact duplicates | Tests covering identical lines - remove duplicates |
| Subset duplicates | Test A is subset of Test B - consolidate or keep only superset |
| Similar pairs | >30% overlap - review if necessary |

### Configuration (pyproject.toml)
```toml
[tool.testiq]
threshold = 0.3  # Similarity threshold (30%)
max_duplicates = 0  # Fail if any duplicates found
```

## Ruff PT014 - Duplicate Parametrization

### Purpose
Detects duplicate parameter sets in pytest.mark.parametrize decorators.

### Installation
```bash
pip install ruff
```

### Usage
```bash
ruff check tests/ --select PT014
```

### Fix
```bash
ruff check tests/ --select PT014 --fix
```

## Vulture - Dead Code Detection

### Purpose
Finds unused code (functions, classes, variables).

### Installation
```bash
pip install vulture
```

### Usage
```bash
# Scan entire package
vulture speechlib/ --min-confidence 60

# Scan specific files
vulture speechlib/speechlib.py speechlib/audio_state.py

# Export to JSON
vulture speechlib/ --json > dead_code.json
```

### Confidence Levels
- 60%: Safe for most code
- 80%: Very confident
- 100%: Whitelist-only (definitely unused)

## Import Linter - Architecture Enforcement

### Purpose
Enforces architectural boundaries and detects circular dependencies.

### Installation
```bash
pip install import-linter
```

### Configuration (pyproject.toml)
```toml
[tool.importlinter]
root_package = "speechlib"

[[tool.importlinter.contracts]]
name = "public-api"
type = "public-api"
is_public = true

[[tool.importlinter.contracts]]
name = "no-circular"
type = "no-circular-dependencies"

[[tool.importlinter.contracts]]
name = "preprocessing-isolation"
type = "independent"
containers = [
    "speechlib.convert_to_wav",
    "speechlib.convert_to_mono",
    "speechlib.re_encode",
    "speechlib.resample_to_16k",
    "speechlib.loudnorm",
    "speechlib.enhance_audio",
]
```

### Usage
```bash
import-linter lint
import-linter lint --verbose
```

## Combined Analysis Script

Run all checks:
```bash
python analyze_tests.py --all
```

Script contents:
```python
#!/usr/bin/env python3
import subprocess

def run_cmd(cmd, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    return subprocess.run(cmd, shell=True).returncode == 0

# Generate coverage
run_cmd('pytest tests/ --testiq-output=testiq_coverage.json', 'Test Coverage')

# TestIQ analysis
run_cmd('python -X utf8 -c "from testiq.cli import main; main([\'analyze\', \'testiq_coverage.json\'])"', 'TestIQ')

# Ruff
run_cmd('ruff check tests/ --select PT014', 'Ruff PT014')

# Vulture
run_cmd('vulture speechlib/ --min-confidence 60', 'Vulture')

# Import Linter
run_cmd('import-linter lint', 'Import Linter')
```

## OpenCode Timeout Configuration

Long-running commands may timeout. Configure in `~/.config/opencode/opencode.json`:

```json
{
  "experimental": {
    "OPENCODE_EXPERIMENTAL_BASH_DEFAULT_TIMEOUT_MS": 600000
  }
}
```

## CI/CD Integration

Add to `.github/workflows/test.yml`:
```yaml
- name: Run analysis
  run: |
    pytest tests/ --testiq-output=testiq_coverage.json
    python -X utf8 -c "from testiq.cli import main; main(['analyze', 'testiq_coverage.json'])"
    ruff check tests/ --select PT014
    vulture speechlib/ --min-confidence 60
    import-linter lint
```

## Common Issues

### TestIQ encoding error on Windows
```bash
python -X utf8 -c "from testiq.cli import main; main(['analyze', 'file.json'])"
```

### Import Linter cache issues
```bash
rm -rf .import_linter_cache
import-linter lint
```

### Vulture false positives
Add to `.vultureignore`:
```
# Whitelist dead code that may be used dynamically
speechlib/speechlib.py:Transcriptor
speechlib/audio_state.py:is_wav
```
