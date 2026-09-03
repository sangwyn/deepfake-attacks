import torch, sys
from pathlib import Path
sys.path.insert(0, "/data2/aiattacks/Mingyao-Duan")
from my_attack.detectors import load_detectors
from my_attack.data import DATASETS, build_manifest, load_image_tensor
from my_attack.vida_attack import margin, vida_attack
W=Path("/data2/aiattacks/Mingyao-Duan/my_attack/weights")
dev=torch.device("cuda")
packs=load_detectors(W,dev,use_dct=True,use_vit=True)
vit=packs["vit_b_16"]["branch"]; dct=packs["densenet121_dct"]["branch"]
fakes=[m["path"] for m in build_manifest(DATASETS["celebA"]) if m["cls"]=="fake"][:16]
fails=[]
for p in fakes:
    x,_=load_image_tensor(p,dev)
    d,used=vida_attack(x,packs,dev,steps=80,tail_steps=60)
    adv=(x+d).clamp(0,1)
    mv,md=margin(vit,adv),margin(dct,adv)
    if md>=0:
        fails.append((p,md,mv,used))
        print(f"FAIL {Path(p).name}  DCT margin {md:.3f}  ViT margin {mv:.3f}  total_steps {used}")
print(f"\n{len(fails)} DCT failures after vida+tail")
