"""Train a small shared depthwise convolutional perturbation."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
import lpips
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import CLASS_IDX_REAL, compute_ssim_rgb, load_model, np_to_lpips_tensor
from attacks.dual_pgd import _vit_preprocess, dct_preprocess

def load_batch(names, size, device):
    return torch.stack([
        torch.from_numpy(np.asarray(Image.open(name).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)).copy())
        .permute(2, 0, 1).float() / 255
        for name in names
    ]).to(device)


def metric_pair(name, attacked, index):
    height, width = attacked.shape[-2:]
    original = np.asarray(Image.open(name).convert("RGB").resize((width, height), Image.Resampling.BICUBIC)).copy()
    adv = (attacked[index].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    return original, adv


def evaluate_filter(paths, kernel, kernel_size, eps, vit, dct, lpips_fn, device, batch_size):
    vit_hits = dct_hits = 0
    ssims, lpips_values = [], []
    with torch.no_grad():
        # The detector branches resize internally, so a fixed canvas avoids one
        # model forward per source resolution in the held-out set.
        for start in range(0, len(paths), batch_size):
            names = paths[start:start + batch_size]
            batch = load_batch(names, 256, device)
            attacked = (batch + F.conv2d(batch, kernel, padding=kernel_size // 2, groups=3).clamp(-eps, eps)).clamp(0, 1)
            vit_hits += int((vit(_vit_preprocess(attacked)).argmax(1) == CLASS_IDX_REAL).sum())
            dct_hits += int((dct(dct_preprocess(attacked, log_scale=True, resize_mode="bicubic")).argmax(1) == CLASS_IDX_REAL).sum())
            originals, adversarials = [], []
            for i, name in enumerate(names):
                original, adv = metric_pair(name, attacked, i)
                originals.append(np_to_lpips_tensor(original, device))
                adversarials.append(np_to_lpips_tensor(adv, device))
                ssims.append(compute_ssim_rgb(original, adv))
            lpips_values.extend(lpips_fn(torch.cat(originals), torch.cat(adversarials)).flatten().cpu().tolist())
    n = len(paths)
    return {"images": n, "vit_success": vit_hits / n, "dct_success": dct_hits / n, "mean_ssim": float(np.mean(ssims)), "mean_lpips": float(np.mean(lpips_values))}


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(cfg["split_json"]).read_text())
    train_paths = split[cfg.get("train_split", "tuning")]
    eval_paths = split[cfg.get("eval_split", "held_out")]
    seed = int(cfg.get("seed", 0)); random.seed(seed); torch.manual_seed(seed)
    models = Path(cfg["models_dir"])
    vit = load_model("vit_b_16", models / "vit_b_16.pth", device)
    dct = load_model("densenet121_dct", models / "densenet121_dct.pth", device)
    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    size = 256; kernel_size = int(cfg.get("kernel_size", 3)); eps = float(cfg.get("epsilon", 8 / 255)); step = float(cfg.get("step_size", 0.5 / 255))
    kernel = torch.zeros((3, 1, kernel_size, kernel_size), device=device, requires_grad=True)
    history = []
    for epoch in range(int(cfg.get("epochs", 8))):
        order = list(train_paths); random.Random(seed + epoch).shuffle(order); losses=[]
        for start in range(0, len(order), int(cfg.get("batch_size", 4))):
            names=order[start:start+int(cfg.get("batch_size",4))]
            batch = load_batch(names, size, device)
            delta=F.conv2d(batch,kernel,padding=kernel_size//2,groups=3).clamp(-eps,eps)
            attacked=(batch+delta).clamp(0,1); target=torch.zeros(len(names),dtype=torch.long,device=device)
            loss=0.5*(F.cross_entropy(vit(_vit_preprocess(attacked)),target)+F.cross_entropy(dct(dct_preprocess(attacked,log_scale=True,resize_mode="bicubic")),target))
            grad=torch.autograd.grad(loss,kernel)[0]
            kernel=(kernel-step*grad.sign()).clamp(-eps,eps).detach().requires_grad_(True); losses.append(float(loss))
        history.append({"epoch":epoch+1,"loss":float(np.mean(losses))}); print(json.dumps(history[-1]),flush=True)
    result={"train_split":cfg.get("train_split","tuning"),"eval_split":cfg.get("eval_split","held_out"),"images":len(eval_paths),"kernel_size":kernel_size,"epsilon":eps,"step_size":step,"iterations":int(cfg.get("epochs",8)),"history":history}
    result.update(evaluate_filter(eval_paths, kernel, kernel_size, eps, vit, dct, lpips_fn, device, int(cfg.get("batch_size",4))))
    Path(cfg["checkpoint"]).parent.mkdir(parents=True,exist_ok=True); torch.save({"kernel":kernel.cpu()},cfg["checkpoint"]); Path(cfg["output_json"]).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); a=p.parse_args(); main(yaml.safe_load(Path(a.config).read_text()))
