"""Entrypoint for the warehouse AGV A* RWARE demo."""

from __future__ import annotations

import os
import sys

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.run_agv_intersection_demo import main


if __name__ == "__main__":
    main()
