import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
os.environ["PACKAGE_ROOT"] = str(PACKAGE_ROOT)
