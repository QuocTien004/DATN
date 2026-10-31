#!/usr/bin/env python
"""Launch the read-only Vietnamese DATN interface from the repository root."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo_ui.app import main

if __name__=='__main__':
    main()
