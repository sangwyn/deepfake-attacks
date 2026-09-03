import torch, sys
from pathlib import Path
sys.path.insert(0, "/data2/aiattacks/Mingyao-Duan")
from my_attack.detectors import load_detectors
from my_attack.data import DATASETS, build_manifest, load_image_tensor
from my_attack.attacks import targeted_margin_loss
from my_attack.masks import region_masks
from my_attack.vida_attack import fullres, route_blind, margin
eps=8/255; alpha=2/255; mu=1.0
W=Path("/data2/aiattacks/Mingyao-Duan/my_attack/weights")
dev=torch.device("cuda")
packs=load_detectors(W,dev,use_dct=True,use_vit=True)
vit=packs["vit_b_16"]["branch"]; dct=packs["densenet121_dct"]["branch"]
p=[m["path"] for m in build_manifest(DATASETS["celebA"]) if m["cls"]=="fake"][0]
x,_=load_image_tensor(p,dev); n,_,h,w=x.shape
rm=region_masks(device=dev)
vitm=fullres(rm["annulus"]+rm["center"],h,w,dev); annm=fullres(rm["annulus"],h,w,dev)
cenm=fullres(rm["center"],h,w,dev); border=fullres(rm["border"],h,w,dev)
d=torch.zeros_like(x); av=torch.zeros_like(x); ad=torch.zeros_like(x)
for it in range(80):
    d=d.detach().requires_grad_(True)
    advv=(x+d).clamp(0,1); advd=(x+d).clamp(0,1)
    gv=torch.autograd.grad(targeted_margin_loss(vit(advv),0),d)[0]
    gd=torch.autograd.grad(targeted_margin_loss(dct(advd),0),d)[0]
    gv=gv/(gv.abs().mean()+1e-12); gd=gd/(gd.abs().mean()+1e-12)
    av=mu*av+gv; ad=mu*ad+gd
    with torch.no_grad():
        sv=route_blind(av.sign(),vitm,annm)
        sd=gd.sign()*cenm  # NOTE: use raw gd sign (not momentum ad) for DCT test
        d=d-alpha*(sv+sd)
        mabs=d.abs().amax()
        if mabs>eps: d=d*(eps/mabs)
    if it%20==0 or it==79:
        a=(x+d).clamp(0,1)
        # center grayscale magnitude of delta (what DCT sees)
        wgt=torch.tensor([0.299,0.587,0.114],device=dev).view(1,3,1,1)
        gray=(d*wgt).sum(1,keepdim=True)
        cgray=gray[:,:,256:768,256:768].abs().mean().item()
        chroma=(d-(d*wgt).sum(1,keepdim=True)*wgt).abs().mean().item()
        print(f"step {it:3d}  ViT {margin(vit,a):7.3f}  DCT {margin(dct,a):7.3f}  "
              f"|d|max {d.abs().amax():.4f}  centerGray {cgray:.4f}  chroma {chroma:.4f}")
