import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

models_dir = Path(__file__).resolve().parent.parent / 'models'
print("Models dir:", models_dir)

try:
    from predict import ChurnPredictor
    pred = ChurnPredictor(str(models_dir))
    print("SUCCESS: ChurnPredictor loaded! Loaded status:", pred.loaded)
except Exception as e:
    print("EXCEPT:", e)
    traceback.print_exc()
