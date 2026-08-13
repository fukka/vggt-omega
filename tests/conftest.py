# Copyright (c) 2026.
"""Put the repo root on ``sys.path`` so both experiments can be inspected.

This suite imports neither of them — it reads their source — but the packages'
own ``conftest`` files do the same thing, and a root suite that cannot be run
from another directory is a root suite nobody runs.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
