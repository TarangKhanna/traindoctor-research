#!/usr/bin/env python3
"""Every number in the README Results section must also appear in docs/EVIDENCE_MAP.md, so no figure is cited
without a recount entry. Numeric values are recomputed from sources by scripts/check_readme.py.
Usage: python scripts/check_readme_numbers.py (repo root). Exits non-zero on any missing number."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
norm = lambda t: t.replace("−", "-")
readme = norm((ROOT / "README.md").read_text())
emap = norm((ROOT / "docs/EVIDENCE_MAP.md").read_text())
sec = readme.split("## Results", 1)[1].split("\n## ", 1)[0]
sec = re.sub(r"\]\([^)]*\)", "]", sec)  # drop link targets
sec = re.sub(r"!\[[^\]]*\]", "", sec)  # drop figure alt text
nums = sorted({m.lstrip("+-") for m in re.findall(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?![\w])", sec)})
missing = [n for n in nums if not re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", emap)]
print(f"{len(nums)} distinct numbers in README Results; missing from EVIDENCE_MAP.md: {missing or 'none'}")
sys.exit(1 if missing else 0)
