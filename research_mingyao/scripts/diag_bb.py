import sys
from pathlib import Path
import torch

REF = Path("/data2/aiattacks/Mingyao-Duan/my_attack/refcode")
sys.path.insert(0, str(REF))
import detectors as Odet
from my_attack.data import DATASETS, build_manifest, load_image_tensor

BW = Path("/data2/aiattacks/Mingyao-Duan/my_attack/third_weights")
ARCH = REF / "deepfakes_code"
dev = torch.device("cuda")

for name in ["npr", "aide"]:
    det = Odet.load_detector(name, BW / f"{name}.pth", dev, architecture_root=ARCH)
    print(f"\n===== {name} =====")
    for cls in ("fake", "real"):
        items = [m for m in build_manifest(DATASETS["celebA"]) if m["cls"] == cls][:4]
        for m in items:
            x, _ = load_image_tensor(m["path"], dev)
            with torch.no_grad():
                logit = det(x)          # adapter maps to [Real, Fake]
            pred = logit.argmax(1).item()
            print(f"  {cls:4s} {Path(m['path']).name[:20]:20s} logits="
                  f"{logit[0].detach().cpu().numpy().round(3)}  pred={'Real' if pred==0 else 'Fake'}")
