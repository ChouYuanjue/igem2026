# Mammoth upstream notice

`mammoth_lwf.py` extracts the generic teacher-student soft-target distillation primitive used by Mammoth continual-learning methods.

- Repository: `https://github.com/aimagelab/mammoth`
- Upstream commit: `e75a491c69fd729edeb01431afb753d9157d9a81`
- Relevant sources: `models/lwf.py`, `models/zscl.py`
- License: MIT; retained as `MAMMOTH_LICENSE`.

Catalyst supplies retrieval logits over the exact legacy candidate universe; no Mammoth classifier/task harness is copied.
