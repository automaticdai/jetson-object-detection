import sys
from pathlib import Path

# Scripts in src/ import each other by bare module name, as they do when run as `python3 src/<script>.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
