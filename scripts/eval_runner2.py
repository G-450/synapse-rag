"""
Wrapper that patches get_qdrant_client to use qdrant_db_eval2,
then delegates to the real evaluation script.
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
python_dir = root_dir / "python"
eval_db = python_dir / "qdrant_db_eval2"

for p in [str(python_dir), str(root_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from qdrant_client import QdrantClient
import app.rag as rag

_eval_client = None

def patched_get_qdrant_client():
    global _eval_client
    if _eval_client is None:
        _eval_client = QdrantClient(path=str(eval_db))
    return _eval_client

rag.get_qdrant_client = patched_get_qdrant_client
rag._qdrant_client = None
rag._vectorstore = None

script = sys.argv[1]
sys.argv = sys.argv[1:]

with open(root_dir / "scripts" / script, "r") as f:
    code = f.read()

exec(compile(code, root_dir / "scripts" / script, "exec"), {"__name__": "__main__", "__file__": str(root_dir / "scripts" / script)})
