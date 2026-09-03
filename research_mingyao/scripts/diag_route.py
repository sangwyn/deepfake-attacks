import torch, sys
from pathlib import Path
sys.path.insert(0, "/data2/aiattacks/Mingyao-Duan")
from my_attack.detectors import load_detectors
from my_attack.data import DATASETS, build_manifest, load_image_tensor
from my_attack.attacks import targeted_margin_loss
from my_attack.masks import region_masks
from my_attack.vida_attack import fullres, route_dct_grayscale, margin, route_center
import torch.nn.functional as F

W = Path("/data2/aiattacks/Mingyao-Duan/my_attack/weights")
dev = torch.device("cuda")
packs = load_detectors(W, dev, use_dct=True, use_vit=False)
dct_b = packs["densenet121_dct"]["branch"]
fakes = [m["path"] for m in build_manifest(DATASETS["celebA"]) if m["cls"]=="fake"][:6]
eps=8/255; alpha=2/255; mu=1.0

def run(kind, steps=80):
    fl=0
    for p in fakes:
        x,_=load_image_tensor(p,dev); n,_,h,w=x.shape
        rm=region_masks(device=dev); cenm=fullres(rm["center"],h,w,dev)
        cenm3=cenm.expand(1,3,h,w)
        d=torch.zeros_like(x); acc=torch.zeros_like(x)
        for it in range(steps):
            d=d.detach().requires_grad_(True)
            adv=(x+d).clamp(0,1)
            g=torch.autograd.grad(targeted_margin_loss(dct_b(adv),0),d)[0]
            g=g/(g.abs().mean()+1e-12); acc=mu*acc+g
            with torch.no_grad():
                if kind=="gray_route":
                    step=route_dct_grayscale(acc,cenm)
                elif kind=="luma_route":
                    step=route_center(acc.sign(),cenm)
                elif kind=="center_fullrgb":
                    step=acc.sign()*cenm3
                else:
                    step=acc.sign()
                d=d-alpha*step
                mm=d.abs().amax()
                if mm>eps: d=d*(eps/mm)
        fl += int(margin(dct_b,(x+d).clamp(0,1))<0)
    print(f"  {kind:14s} DCT flip {fl}/{len(fakes)}")

for k in ("fullsign","center_fullrgb","gray_route","luma_route"):
    run(k)
