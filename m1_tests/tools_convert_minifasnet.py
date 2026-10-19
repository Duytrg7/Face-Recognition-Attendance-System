"""Export Silent-Face MiniFASNet .pth -> ONNX (softmax included).
Run from inside the Silent-Face-Anti-Spoofing repo root:  python convert_minifasnet.py
Input : 1x3x80x80, BGR, float32 in [0,255] (no normalization, same as original repo)
Output: 1x3 softmax probabilities; index 1 = real face.
"""
import os, sys, torch, torch.nn as nn
sys.path.insert(0, os.getcwd())
from src.anti_spoof_predict import MODEL_MAPPING
from src.utility import get_kernel, parse_model_name
from collections import OrderedDict

class WithSoftmax(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return torch.softmax(s.m(x), dim=1)

src_dir = "resources/anti_spoof_models"; out_dir = "onnx_models"; os.makedirs(out_dir, exist_ok=True)
for name in sorted(os.listdir(src_dir)):
    if not name.endswith(".pth"): continue
    h, w, mtype, scale = parse_model_name(name)
    model = MODEL_MAPPING[mtype](conv6_kernel=get_kernel(h, w))
    sd = torch.load(os.path.join(src_dir, name), map_location="cpu")
    sd = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in sd.items())
    model.load_state_dict(sd); model.eval()
    out = os.path.join(out_dir, name.replace(".pth", ".onnx"))
    torch.onnx.export(WithSoftmax(model), torch.zeros(1, 3, h, w), out,
                      input_names=["input"], output_names=["prob"], opset_version=13, dynamo=False)
    print("exported", out, "scale", scale)
