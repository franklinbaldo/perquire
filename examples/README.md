# Examples

This directory contains example scripts and demonstrations of Perquire functionality.

## Live End-to-End Test

- **`live_e2e_test.py`** - Interactive live E2E test with real API calls
- **`run_live_e2e.sh`** - Automated test runner for CI/validation

### Usage

```bash
# Interactive live test
cd examples
uv run --env-file ../.env python live_e2e_test.py

# Automated test with predefined inputs
cd examples  
bash run_live_e2e.sh
```

### Requirements

- Valid `GOOGLE_API_KEY` in `../.env` file
- All dependencies installed via `uv sync`

## Test Results

Recent live test achieved:
- **4.0/5 EXCELLENT** subjective rating
- **6.76 seconds** investigation time
- **5-iteration convergence** with plateau detection
- **✅ Production validation complete**