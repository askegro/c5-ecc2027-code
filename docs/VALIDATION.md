# Numerical checks

## Commands

```bash
python run.py check
python run.py test
python run.py report
```

- `check` parses Python source and checks required reference inputs.
- `test` exercises the CGE numerical tests: exact recovery, determinism, prediction with offsets and noise, scale invariance, and degenerate inputs.
- `report` runs 17 analysis and reporting steps and checks generated TeX files for NaN values.

## Trace archive checks

The complete main-study archive matches all 576 paths in `code/confirm.list`, with 96 units and six records each. No missing, extra, or duplicate paths were found. Gzip integrity passed. See [trace inputs](TRACE_REPRODUCTION.md) for the archive checksum and verification command.
