#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.plot_results import main


if __name__ == "__main__":
    print("[info] summarize_results.py now only plots cached results. Run cache_results.py first if needed.")
    main()
