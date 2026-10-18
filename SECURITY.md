# Security policy

This repository holds research artifacts: training and analysis code, recorded results and documentation. The verifier, sandbox backends and trust boundary are in the [traindoctor-env](https://github.com/TarangKhanna/traindoctor-env) library, and its `SECURITY.md` applies.

Report suspected verifier bypasses or secret exposure through a GitHub private security advisory (Security tab, "Report a vulnerability"), not a public issue. Scripts here that execute agent-proposed repairs (`scripts/redteam_suite.py`, `scripts/redteam_a2v2.py`) must only be run against a disposable sandbox.

This is research code without an external security review.
