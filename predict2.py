import os
import torch
import numpy as np
from model2 import TreatmentResponseModel

_device = torch.device("cpu")
_model = None
_pack = None

def _load_model():
    global _model, _pack
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    ckpt = os.path.join(here, "stage2_model.pth")
    obj = torch.load(ckpt, map_location=_device)

    feature_cols = obj["feature_cols"]
    mean = obj["mean"]
    std = obj["std"]

    _model = TreatmentResponseModel(in_dim=10).to(_device)
    _model.load_state_dict(obj["state_dict"], strict=True)
    _model.eval()

    _pack = {"feature_cols": feature_cols, "mean": mean, "std": std}

def predict_response_from_10(gene10):
    _load_model()

    x = np.array(gene10, dtype=np.float32)

    mu = np.array([_pack["mean"][c] for c in _pack["feature_cols"]], dtype=np.float32)
    sd = np.array([_pack["std"][c] for c in _pack["feature_cols"]], dtype=np.float32)
    sd[sd == 0] = 1.0
    x = (x - mu) / sd

    xt = torch.tensor(x, dtype=torch.float32, device=_device).unsqueeze(0)

    with torch.no_grad():
        logits = _model(xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    p0 = float(probs[0])  # Not respond
    p1 = float(probs[1])  # Respond
    debug = f"Predicted class={pred} | P0={p0:.2f}, P1={p1:.2f} (0=Not Respond, 1=Respond)"
    return pred, p0, p1, debug
