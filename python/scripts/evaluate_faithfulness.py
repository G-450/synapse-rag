import sys
from pathlib import Path

# Forward to scripts/evaluate_faithfulness.py
root_script = Path(__file__).resolve().parent.parent.parent / "scripts" / "evaluate_faithfulness.py"
if root_script.exists():
    with open(root_script, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, str(root_script), "exec"))
else:
    print(f"Cannot find root script at {root_script}")
