# Contributing

For a reproducibility report, include your Git commit, command, dataset preparation, GPU/PyTorch/CUDA versions, checkpoint hashes, and a minimal log excerpt. Redact local account details and credentials.

For code changes, keep the experimental protocol explicit and distinguish RTTS refinement from gradient-based adaptation. Run `python -m unittest discover -s tests -p "test_*.py"` and `python -m compileall -q adapt ovss utils_local sam2 scripts main.py` before opening a pull request. Full evaluation changes should include per-corruption metrics and a description of the evaluation split.
