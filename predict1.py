import os
import torch
import numpy as np

from model import DiseaseModel

_device = torch.device("cpu")
_pack = None
_model = None


def _load():
    global _pack, _model
    if _model is not None:
        return

    here = os.path.dirname(__file__)
    ckpt = os.path.join(here, "stage1_model.pth")
    obj = torch.load(ckpt, map_location=_device)

    classes = obj["classes"]
    feature_cols = obj["feature_cols"]
    mean = obj["mean"]
    std = obj["std"]

    _model = DiseaseModel(in_dim=10, num_classes=len(classes)).to(_device)
    _model.load_state_dict(obj["state_dict"], strict=True)
    _model.eval()

    _pack = {"classes": classes, "feature_cols": feature_cols, "mean": mean, "std": std}


def predict_disease_from_10(inputs10):
    """
    inputs10 order must match frontend:
    [Age, Gender(0/1), WBC, CRP, ESR, ANA, RF, dsDNA, IL6, TNF]
    Returns: pred_id, pred_label, confidence, probs_dict
    """
    _load()

    x = np.array(inputs10, dtype=np.float32)

    # apply same normalization used during training
    mu = np.array([_pack["mean"][c] for c in _pack["feature_cols"]], dtype=np.float32)
    sd = np.array([_pack["std"][c] for c in _pack["feature_cols"]], dtype=np.float32)
    sd[sd == 0] = 1.0
    x = (x - mu) / sd

    xt = torch.tensor(x, dtype=torch.float32, device=_device).unsqueeze(0)

    with torch.no_grad():
        logits = _model(xt)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_label = _pack["classes"][pred_id]
    confidence = float(probs[pred_id])

    probs_dict = { _pack["classes"][i]: float(probs[i]) for i in range(len(_pack["classes"])) }
    return pred_id, pred_label, confidence, probs_dict
