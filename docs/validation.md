# Release validation

Checks performed on 2026-10-09:

- SHA-256 consistency for all 130 retained anonymous source/assets files, with text line endings normalized.
- Syntax parsing of every released Python source.
- Launcher help without machine-learning dependencies.
- Seven preset command builders preserving the anonymous `mlmp --adapt` setting.
- Clean/corruption aggregation, including incomplete-suite, duplicate, and empty-input handling.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

These release checks do not execute SAM or reproduce benchmark mIoU. The available local environment lacks the full documented evaluation stack and SAM checkpoint. No packages were installed or environments changed during preparation.
