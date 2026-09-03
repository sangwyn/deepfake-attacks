import sys
from pathlib import Path
import torch

REF = Path("/data2/aiattacks/Mingyao-Duan/my_attack/refcode")
sys.path.insert(0, str(REF))
sys.path.insert(0, str(REF))
import detectors as Odet

W = Path("/data2/aiattacks/Mingyao-Duan/my_attack/third_weights")
ARCH = REF / "deepfakes_code"
dev = torch.device("cuda")

for name in ["vit_b_16", "densenet121_dct", "npr", "aide"]:
    det = Odet.load_detector(name, W / f"{name}.pth", dev, architecture_root=ARCH)
    print(f"[loaded] {name:16s} OK")
print("ALL FOUR DETECTORS LOADED")
