# Changelog

## 0.1.0

- Import the anonymous RTTS implementation with source provenance retained.
- Provide an explicit frozen-model RTTS evaluator and two-round refinement.
- Process sliding-window patches individually for SAM proposal generation.
- Make SAM checkpoints configurable and freeze the proposal model.
- Separate baseline execution from explicitly requested archived combinations.
- Add seven portable benchmark presets, environment checks, checkpoint download, and complete-corruption summarization.
- Correct the saved command entry point and seed handling; use stable validation ordering.
- Replace the workstation dependency export with direct dependencies and pinned SAM installation.
- Add method documentation, dataset layouts, manuscript citation, third-party license texts, and release checks.
- Document unresolved differences between historical optimizer settings and the manuscript's training-free protocol.
