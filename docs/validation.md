# Release validation

Validation performed on 2026-10-09, before public release.

| Check | Outcome |
| :--- | :--- |
| Python source syntax | Passed for all released Python files |
| CLI help without ML packages | Passed |
| Training-free CLI rejects `--adapt` | Passed |
| All seven preset command builders | Passed; no optimizer options in RTTS commands |
| Result summarization | Passed; corruption average requires all 15 types and excludes clean score |
| Empty/duplicate result rejection | Passed |
| Extracted Sinkhorn vs archived normalization | Passed on real PyTorch tensors |
| Empty-mask pooling and no-proposal fallback | Passed |
| Two feedback rounds and dynamic grid | Passed on synthetic proposals |
| Tiny actual NaCLIP forward through RTTS wrapper | Passed; two patches returned expected shapes; all weights frozen and unchanged |

Commands:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The complete 12-test suite passed in the existing `pytorch` environment (PyTorch 1.13.1+cu116). Lightweight checks also passed with the system Python, where tensor tests were skipped because PyTorch was absent. GitHub Actions runs the tensor regression job with CPU PyTorch 2.1.2. Passing these checks does not certify full benchmark scores.

## End-to-end limitation

The local environment lacked `mmcv`, `mmseg`, and `segment_anything`; its PyTorch version also differed from the documented evaluation stack. The release preparation did not install packages or modify the existing conda environment. A real SAM checkpoint, complete prepared validation datasets, and the reference stack are required to verify mIoU and timings. Full SAM inference, the manuscript table, and optional SAM2/SAM3/CorrCLIP experiments have not been rerun for this release.
