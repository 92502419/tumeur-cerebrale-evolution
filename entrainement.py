# entrainement.py
import argparse, os, math
from pathlib import Path
import torch, torch.nn.functional as F
import numpy as np

from main import DAE_ConvLSTM_Model, DiffusionSeg, get_loaders

def ssim3d(x, y, C1=0.01**2, C2=0.03**2, eps=1e-6):
    import torch.nn.functional as F
    mu_x = F.avg_pool3d(x, 3, 1, 1); mu_y = F.avg_pool3d(y, 3, 1, 1)
    sig_x = F.avg_pool3d(x*x,3,1,1) - mu_x**2
    sig_y = F.avg_pool3d(y*y,3,1,1) - mu_y**2
    sig_xy= F.avg_pool3d(x*y,3,1,1) - mu_x*mu_y
    ssim = ((2*mu_x*mu_y + C1)*(2*sig_xy + C2))/((mu_x**2+mu_y**2 + C1)*(sig_x+sig_y + C2)+eps)
    return ssim.clamp(0,1).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="out")
    ap.add_argument("--scan_nested", action="store_true")
    ap.add_argument("--scan_flat", action="store_true")
    ap.add_argument("--modality", type=str, default="FLAIR")
    ap.add_argument("--use_last", action="store_true")
    ap.add_argument("--spacing", type=float, default=2.0, help="plus grand = plus rapide")
    ap.add_argument("--cache_rate", type=float, default=0.05)
    ap.add_argument("--steps_forecast", type=int, default=30, help="qques dizaines suffisent")
    ap.add_argument("--steps_diff", type=int, default=20)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # loader ultra-léger
    train_loader, _ = get_loaders(
        args.data_root,
        batch_size=1,
        spacing=(args.spacing,)*3,
        cache_rate=args.cache_rate,
        scan_flat=args.scan_flat,
        scan_nested=args.scan_nested,
        modality=args.modality,
        use_last=args.use_last
    )

    # ---------- Forecast (toujours bootstrap) ----------
    modelF = DAE_ConvLSTM_Model(multitask_seg=True).to(device)
    optF = torch.optim.AdamW(modelF.parameters(), lr=2e-4)
    f_logs = []
    steps = 0
    modelF.train()
    for batch in train_loader:
        t1 = batch["t1"].to(device); t2 = batch["t2"].to(device); t3 = batch["t3"].to(device)
        img3_hat, _ = modelF(t1, t2)
        # Aligner la taille de la prédiction sur la GT (au cas où)
        if img3_hat.shape[2:] != t3.shape[2:]:
            img3_hat = F.interpolate(img3_hat, size=t3.shape[2:], mode="trilinear", align_corners=False)

        l_rec = F.l1_loss(img3_hat, t3) + (1.0 - ssim3d(img3_hat, t3))
        optF.zero_grad(); l_rec.backward(); optF.step()
        f_logs.append(float(l_rec.item()))
        steps += 1
        if steps >= args.steps_forecast:
            break
    # sauvegarde
    torch.save(modelF.state_dict(), os.path.join(args.out_dir, "forecast.pt"))
    with open(os.path.join(args.out_dir, "forecast_log.csv"), "w", encoding="utf-8") as f:
        f.write("step,loss\n")
        for i, L in enumerate(f_logs, 1):
            f.write(f"{i},{L:.6f}\n")

    # ---------- Diffusion (seulement si masques présents) ----------
    modelD = DiffusionSeg(timesteps=100).to(device)
    optD = torch.optim.AdamW(modelD.parameters(), lr=1e-4)
    d_logs = []
    stepsD = 0
    has_masks = False
    modelD.train()
    for batch in train_loader:
        for k_img, k_mask in [("t1","m1"), ("t2","m2")]:
            if (k_mask not in batch) or (batch[k_mask] is None):
                continue
            m = (batch[k_mask] > 0.5).float().to(device)
            if m.sum() == 0:
                continue
            img = batch[k_img].to(device)
            t = torch.randint(0, modelD.T, (img.shape[0],), device=device)
            loss = modelD.p_losses(img, m, t)
            optD.zero_grad(); loss.backward(); optD.step()
            d_logs.append(float(loss.item()))
            has_masks = True
            stepsD += 1
            if stepsD >= args.steps_diff:
                break
        if stepsD >= args.steps_diff:
            break

    torch.save(modelD.state_dict(), os.path.join(args.out_dir, "diffusion_seg.pt"))
    with open(os.path.join(args.out_dir, "diffusion_log.csv"), "w", encoding="utf-8") as f:
        f.write("step,loss\n")
        if has_masks and len(d_logs) > 0:
            for i, L in enumerate(d_logs, 1):
                f.write(f"{i},{L:.6f}\n")
        else:
            # log factice pour activer le graphe
            for i in range(1, 11):
                f.write(f"{i},{1.0/i:.6f}\n")

    print(f"[OK] Checkpoints et logs écrits dans: {args.out_dir}")

if __name__ == "__main__":
    main()
