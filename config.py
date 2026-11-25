"""
Project configuration and path resolution.
Defines absolute paths for data and cache directories.
"""

import sys
from pathlib import Path

# Absolute resolution of the project root directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent

# Main directory paths
DATA_DIR: Path = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
CACHE_DIR: Path = PROJECT_ROOT / ".cache"

# Dynamically add the project root to PYTHONPATH to fix ModuleNotFoundError imports
sys.path.append(str(PROJECT_ROOT))