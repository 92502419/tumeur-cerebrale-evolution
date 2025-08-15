# -*- coding: utf-8 -*-
"""
Projet Tumor Evolution — Segmentation (Diffusion) + Prédiction (DAE + ConvLSTM3D)
- Compatible 3 mises en page des données :
  (A) Dossiers patients "classiques": data/patient_xxx/t1.t2.t3(.nii.gz)
  (B) Fichiers "plats" nommés PATIENT_YYYY-MM-DD_HH-MM-SS_MODALITY.nii(.gz)  -> --scan_flat
  (C) Dossiers imbriqués <root>/<PID>/<YYYY-MM-DD>/<PID>_<date>_<time>_<MOD>.nii.gz -> --scan_nested (ton cas)

Exécution typique (Windows PowerShell) avec jeu "Yale-Brain-Mets-Longitudinal":
python main.py --scan_nested --modality FLAIR --data_root "G:\PKG - Yale-Brain-Mets-Longitudinal\Yale-Brain-Mets-Longitudinal" --train_forecast --batch_size 1 --cache_rate 0.1 --forecast_epochs 10

Dépendances: torch, monai, nibabel, tqdm, einops
"""

import os, sys, math, argparse, json, time, re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from monai.transforms import DivisiblePadd
import nibabel as nib

import monai
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped
)
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss


# =========================
# Utils & I/O
# =========================
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)

def save_nii_like(ref_path, array_ch_first, out_path):
    """Sauve un tenseur (C=1) au format NIfTI, en reprenant affine/header du ref_path."""
    if isinstance(array_ch_first, torch.Tensor):
        array_ch_first = array_ch_first.detach().cpu().numpy()
    vol = np.squeeze(array_ch_first, axis=0)
    ref = nib.load(ref_path)
    img = nib.Nifti1Image(vol.astype(np.float32), affine=ref.affine, header=ref.header)
    nib.save(img, out_path)

def device_auto():
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Helpers dataset (parsing)
# =========================
def _is_nii(p):
    s = str(p).lower()
    return s.endswith(".nii") or s.endswith(".nii.gz")

def _parse_name_flat(path):
    """
    Parse un nom de fichier du type: PID_YYYY-MM-DD_HH-MM-SS_MODALITY.nii(.gz)
    Retourne dict {pid, dt, mod, is_mask} ou None si non conforme
    """
    name = Path(path).name
    base = name[:-7] if name.lower().endswith(".nii.gz") else name[:-4] if name.lower().endswith(".nii") else name
    toks = base.split("_")
    date_idx = None
    for i, t in enumerate(toks):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
            date_idx = i
            break
    if date_idx is None:
        return None
    pid = "_".join(toks[:date_idx]) or "UNKNOWN"
    date_tok = toks[date_idx]
    time_tok = toks[date_idx+1] if date_idx+1 < len(toks) and re.fullmatch(r"\d{2}-\d{2}-\d{2}", toks[date_idx+1]) else "00-00-00"
    mod_tok  = toks[date_idx+2].upper() if date_idx+2 < len(toks) else "UNK"
    try:
        dt = datetime.strptime(f"{date_tok} {time_tok.replace('-',':')}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        dt = datetime.strptime(date_tok, "%Y-%m-%d")
    is_mask = any(k in base.lower() for k in ["mask", "seg", "label"])
    return {"pid": pid, "dt": dt, "mod": mod_tok, "path": str(path), "is_mask": is_mask}

def _parse_name_nested(pid, date_dir, filename):
    """
    Cas imbriqué: <root>/<PID>/<YYYY-MM-DD>/<PID>_<YYYY-MM-DD>_<HH-MM-SS>_<MOD>.nii.gz
    On se base sur filename si possible; sinon sur le nom du dossier date.
    """
    base = filename
    if base.lower().endswith(".nii.gz"):
        base = base[:-7]
    elif base.lower().endswith(".nii"):
        base = base[:-4]
    toks = base.split("_")
    # Cherche la date dans filename; sinon dossier
    date_tok = None
    time_tok = "00-00-00"
    mod_tok = "UNK"
    for i, t in enumerate(toks):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
            date_tok = t
            # time
            if i+1 < len(toks) and re.fullmatch(r"\d{2}-\d{2}-\d{2}", toks[i+1]):
                time_tok = toks[i+1]
                if i+2 < len(toks):
                    mod_tok = toks[i+2].upper()
            elif i+1 < len(toks):
                mod_tok = toks[i+1].upper()
            break
    if date_tok is None:
        # fallback: dossier date
        date_tok = date_dir
        # modalité: dernier token du filename si plausible
        if len(toks) >= 1:
            mod_tok = toks[-1].upper()
    try:
        dt = datetime.strptime(f"{date_tok} {time_tok.replace('-',':')}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        dt = datetime.strptime(date_tok, "%Y-%m-%d")
    is_mask = any(k in filename.lower() for k in ["mask", "seg", "label"])
    return {"pid": pid, "dt": dt, "mod": mod_tok, "is_mask": is_mask}

# =========================
# Builders d'items
# =========================
def build_items_classic(root):
    """
    Données au format:
      data/patient_xxx/t1.nii.gz t2.nii.gz t3.nii.gz (masques optionnels)
    """
    items = []
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dossier data introuvable: {root}")
    for pdir in sorted(root.iterdir()):
        if not pdir.is_dir():
            continue
        t1 = pdir / "t1.nii.gz"
        t2 = pdir / "t2.nii.gz"
        t3 = pdir / "t3.nii.gz"
        if not (t1.exists() and t2.exists() and t3.exists()):
            # essaie .nii
            t1 = pdir / "t1.nii"
            t2 = pdir / "t2.nii"
            t3 = pdir / "t3.nii"
            if not (t1.exists() and t2.exists() and t3.exists()):
                continue
        def opt(p): return str(p) if p.exists() else None
        items.append({
            "t1": str(t1), "t2": str(t2), "t3": str(t3),
            "m1": opt(pdir/"mask_t1.nii.gz") or opt(pdir/"mask_t1.nii"),
            "m2": opt(pdir/"mask_t2.nii.gz") or opt(pdir/"mask_t2.nii"),
            "m3": opt(pdir/"mask_t3.nii.gz") or opt(pdir/"mask_t3.nii"),
            "ref": str(t1)
        })
    if not items:
        raise RuntimeError("Aucun patient valide (format classique).")
    return items

def build_items_flat(root, modality="FLAIR", use_last=False):
    """
    Fichiers "plats" sous root:
      PID_YYYY-MM-DD_HH-MM-SS_MODALITY.nii[.gz]
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dossier data introuvable: {root}")
    modality = (modality or "FLAIR").upper()

    entries = []
    for p in root.rglob("*"):
        if p.is_file() and _is_nii(p):
            info = _parse_name_flat(p)
            if info: entries.append(info)
    if not entries:
        raise RuntimeError("Aucun NIfTI trouvé (scan plat).")

    # group by PID
    by_pid = {}
    for e in entries:
        by_pid.setdefault(e["pid"], []).append(e)

    items = []
    for pid, lst in by_pid.items():
        imgs = [e for e in lst if (not e["is_mask"]) and (e["mod"] == modality)]
        masks= [e for e in lst if e["is_mask"]]
        if len(imgs) < 3:
            continue
        imgs.sort(key=lambda x: x["dt"])
        chosen = imgs[-3:] if use_last else imgs[:3]
        chosen.sort(key=lambda x: x["dt"])

        def find_mask(dt):
            # associe masque date exacte si présent
            for m in masks:
                if m["dt"] == dt:
                    return m["path"]
            return None

        t1, t2, t3 = chosen[0]["path"], chosen[1]["path"], chosen[2]["path"]
        m1 = find_mask(chosen[0]["dt"])
        m2 = find_mask(chosen[1]["dt"])
        m3 = find_mask(chosen[2]["dt"])
        items.append({"t1": t1, "t2": t2, "t3": t3, "m1": m1, "m2": m2, "m3": m3, "ref": t1})
    if not items:
        raise RuntimeError("Aucun trio (t1,t2,t3) trouvé en mode plat.")
    return items

def build_items_nested(root, modality="FLAIR", use_last=False):
    """
    Arborescence imbriquée (TON CAS) :
      <root>/<PID>/<YYYY-MM-DD>/<PID>_<YYYY-MM-DD>_<HH-MM-SS>_<MOD>.nii.gz
    Exemple :
      G:\...\Yale-Brain-Mets-Longitudinal\YG_0B4NV6E3KEZQ\2015-09-29\YG_0B4NV6E3KEZQ_2015-09-29_12-36-27_FLAIR.nii.gz
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dossier data introuvable: {root}")
    modality = (modality or "FLAIR").upper()

    entries = []
    # on parcourt: patients -> dates -> fichiers
    for pid_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        pid = pid_dir.name.strip()
        for date_dir in sorted([d for d in pid_dir.iterdir() if d.is_dir()]):
            date_name = date_dir.name.strip()
            for f in sorted([x for x in date_dir.iterdir() if x.is_file() and _is_nii(x)]):
                info = _parse_name_nested(pid, date_name, f.name)
                # filtre modalité
                if info["mod"] != modality:
                    continue
                entries.append({"pid": pid, "dt": info["dt"], "mod": info["mod"], "path": str(f), "is_mask": info["is_mask"]})

    if not entries:
        raise RuntimeError("Aucun NIfTI FLAIR trouvé en mode imbriqué. Vérifie --modality et le chemin --data_root.")

    # group by PID
    by_pid = {}
    for e in entries:
        by_pid.setdefault(e["pid"], []).append(e)

    items = []
    for pid, lst in by_pid.items():
        imgs = [e for e in lst if not e["is_mask"]]
        masks= [e for e in lst if e["is_mask"]]
        if len(imgs) < 3:
            continue
        imgs.sort(key=lambda x: x["dt"])
        chosen = imgs[-3:] if use_last else imgs[:3]
        chosen.sort(key=lambda x: x["dt"])

        # si jamais des masques existent, on les matche à la date exacte
        def find_mask(dt):
            for m in masks:
                if m["dt"] == dt:
                    return m["path"]
            return None

        t1, t2, t3 = chosen[0]["path"], chosen[1]["path"], chosen[2]["path"]
        m1 = find_mask(chosen[0]["dt"])
        m2 = find_mask(chosen[1]["dt"])
        m3 = find_mask(chosen[2]["dt"])
        items.append({"t1": t1, "t2": t2, "t3": t3, "m1": m1, "m2": m2, "m3": m3, "ref": t1})

    if not items:
        raise RuntimeError("Aucun trio (t1,t2,t3) trouvé en mode imbriqué.")
    return items


# =========================
# Transforms / Loaders
# =========================
def get_transforms(spacing=(1.2,1.2,1.2)):
    keys_img = ["t1","t2","t3"]
    keys_mask= ["m1","m2","m3"]
    return monai.transforms.Compose([
        LoadImaged(keys=keys_img+keys_mask, allow_missing_keys=True),
        EnsureChannelFirstd(keys=keys_img+keys_mask, allow_missing_keys=True),
        Orientationd(keys=keys_img+keys_mask, axcodes="RAS", allow_missing_keys=True),
        Spacingd(keys=keys_img+keys_mask, pixdim=spacing, mode=("bilinear"), allow_missing_keys=True),
        ScaleIntensityRanged(keys=keys_img, a_min=0, a_max=99, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys_img+keys_mask, source_key="t1", allow_missing_keys=True),
        # 🔧 Nouveau : pad pour que D,H,W soient divisibles par 8
        DivisiblePadd(keys=keys_img+keys_mask, k=8, mode="constant", allow_missing_keys=True),
        EnsureTyped(keys=keys_img+keys_mask, dtype=torch.float32, allow_missing_keys=True),
    ])


def get_loaders(data_root, batch_size=1, spacing=(1.2,1.2,1.2), cache_rate=0.2,
                scan_flat=False, scan_nested=False, modality="FLAIR", use_last=False):
    if scan_nested:
        items = build_items_nested(data_root, modality=modality, use_last=use_last)
    elif scan_flat:
        items = build_items_flat(data_root, modality=modality, use_last=use_last)
    else:
        items = build_items_classic(data_root)

    # --- PATCH IMPORTANT : supprimer les clés masque None ---
    cleaned = []
    for it in items:
        # enlève m1/m2/m3 si None
        for mk in ("m1", "m2", "m3"):
            if it.get(mk) is None:
                it.pop(mk, None)
        # sécurité : ne garde que les items qui ont bien t1,t2,t3
        if all(k in it for k in ("t1","t2","t3")):
            cleaned.append(it)
    items = cleaned
    # --------------------------------------------------------


    # split train/val simple (80/20)
    n_train = int(0.8 * len(items)) if len(items) > 4 else max(1, len(items) - 1)
    train, val = items[:n_train], items[n_train:]
    tfm = get_transforms(spacing)

    # NOTE: data_root = gros disque -> évite de tout cacher en RAM
    ds_train = CacheDataset(train, transform=tfm, cache_rate=cache_rate, num_workers=0)
    ds_val   = CacheDataset(val,   transform=tfm, cache_rate=min(0.5, cache_rate), num_workers=0)
    return DataLoader(ds_train, batch_size=batch_size, shuffle=True), DataLoader(ds_val, batch_size=1, shuffle=False)


# =========================
# PARTIE 1 : Diffusion pour Segmentation
# =========================
class EpsilonUNet3D(nn.Module):
    """U-Net 3D prédit le bruit epsilon pour le masque, conditionné par l'image."""
    def __init__(self, in_ch=2, out_ch=1):
        super().__init__()
        self.net = UNet(
            spatial_dims=3,
            in_channels=in_ch,       # [image, masque_bruité]
            out_channels=out_ch,     # epsilon
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
        )
    def forward(self, x):
        return self.net(x)

class DiffusionSeg(nn.Module):
    def __init__(self, timesteps=300, beta_start=1e-4, beta_end=0.02, in_ch=2):
        super().__init__()
        self.T = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.eps_net = EpsilonUNet3D(in_ch=in_ch)

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        a_bar = self.alphas_cumprod[t].view(-1,1,1,1,1)
        return a_bar.sqrt()*x0 + (1-a_bar).sqrt()*noise

    def p_losses(self, img_cond, x0_mask, t):
        noise = torch.randn_like(x0_mask)
        x_t = self.q_sample(x0_mask, t, noise)
        eps_pred = self.eps_net(torch.cat([img_cond, x_t], dim=1))
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def sample(self, img_cond, shape, steps=None, device="cpu"):
        T = self.T if steps is None else min(steps, self.T)
        x_t = torch.randn(shape, device=device)
        for t in reversed(range(T)):
            eps = self.eps_net(torch.cat([img_cond, x_t], dim=1))
            beta_t = self.betas[t]
            alpha_t = 1.0 - beta_t
            a_bar_t = self.alphas_cumprod[t]
            coef1 = (1/alpha_t)**0.5
            coef2 = (1 - alpha_t) / (1 - a_bar_t).sqrt()
            x_t = coef1*(x_t - coef2*eps)
            if t > 0:
                x_t = x_t + (beta_t**0.5)*torch.randn_like(x_t)
        return torch.sigmoid(x_t)

def train_diffusion(loader, epochs=20, lr=1e-4, device="cpu", timesteps=300):
    model = DiffusionSeg(timesteps=timesteps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    T = model.T
    has_masks = False
    for epoch in range(epochs):
        model.train()
        losses=[]
        for batch in loader:
            for k_img, k_mask in [("t1","m1"), ("t2","m2")]:
                if k_mask not in batch or batch[k_mask] is None:
                    continue
                img = batch[k_img].to(device)   # (B,1,D,H,W)
                msk = (batch[k_mask]>0.5).float().to(device)
                if msk.numel()==0:
                    continue
                has_masks = True
                t = torch.randint(0, T, (img.shape[0],), device=device, dtype=torch.long)
                loss = model.p_losses(img, msk, t)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        print(f"[Diffusion][{epoch+1}/{epochs}] loss={np.mean(losses) if losses else float('nan'):.4f}")
    if not has_masks:
        print("ATTENTION: aucun masque (m1/m2) trouvé, la diffusion n'a pas été entraînée.")
    return model

@torch.no_grad()
def infer_diffusion(model, batch, device="cpu", thr=0.5, steps=None):
    img = batch["t3"].to(device)
    B,_,D,H,W = img.shape
    prob = model.sample(img, (B,1,D,H,W), steps=steps, device=device)
    seg = (prob>thr).float()
    return seg, prob


# =========================
# PARTIE 2 : DAE 3D + ConvLSTM3D (prévision t3)
# =========================
class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, base=32, z_ch=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base, base, 3, stride=2, padding=1), nn.LeakyReLU(inplace=True),   # /2
            nn.Conv3d(base, base*2, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base*2, base*2, 3, stride=2, padding=1), nn.LeakyReLU(inplace=True),# /4
            nn.Conv3d(base*2, z_ch, 3, padding=1), nn.LeakyReLU(inplace=True)
        )
    def forward(self, x): return self.enc(x)

class Decoder3D(nn.Module):
    def __init__(self, out_ch=1, base=32, z_ch=128):
        super().__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(z_ch, base*2, 2, stride=2), nn.LeakyReLU(inplace=True),   # x2
            nn.Conv3d(base*2, base*2, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(base*2, base, 2, stride=2), nn.LeakyReLU(inplace=True),    # x2
            nn.Conv3d(base, base, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base, out_ch, 1)
        )
    def forward(self, z): return self.dec(z)

class ConvLSTMCell3D(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        p = k//2
        self.conv = nn.Conv3d(in_ch+hid_ch, 4*hid_ch, k, padding=p)
        self.hid_ch = hid_ch
    def forward(self, x, h, c):
        if h is None:
            b,_,d,h_,w = x.shape
            h = torch.zeros(b, self.hid_ch, d, h_, w, device=x.device)
            c = torch.zeros_like(h)
        gates = self.conv(torch.cat([x,h], dim=1))
        i,f,o,g = torch.chunk(gates, 4, dim=1)
        i,f,o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f*c + i*g
        h = o*torch.tanh(c)
        return h,c

class ConvLSTM3D(nn.Module):
    def __init__(self, in_ch, hid_ch, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([ConvLSTMCell3D(in_ch if i==0 else hid_ch, hid_ch) for i in range(num_layers)])
    def forward(self, xs):
        h = [None]*len(self.layers)
        c = [None]*len(self.layers)
        out = None
        for x in xs:
            for i,cell in enumerate(self.layers):
                h[i], c[i] = cell(x if i==0 else h[i-1], h[i], c[i])
            out = h[-1]
        return out

class DAE_ConvLSTM_Model(nn.Module):
    def __init__(self, z_ch=128, hid_ch=128, multitask_seg=True):
        super().__init__()
        self.enc = Encoder3D(z_ch=z_ch)
        self.dec_img = Decoder3D(out_ch=1, z_ch=z_ch)
        self.convlstm = ConvLSTM3D(in_ch=z_ch, hid_ch=hid_ch, num_layers=1)
        self.multitask_seg = multitask_seg
        if multitask_seg:
            self.dec_seg = Decoder3D(out_ch=1, z_ch=hid_ch)
    def forward(self, t1, t2):
        z1 = self.enc(t1)
        z2 = self.enc(t2)
        z3_hat = self.convlstm([z1, z2])
        img3_hat = self.dec_img(z3_hat)
        seg3_hat = torch.sigmoid(self.dec_seg(z3_hat)) if self.multitask_seg else None
        return img3_hat, seg3_hat

def ssim3d(x,y, C1=0.01**2, C2=0.03**2, eps=1e-6):
    mu_x = F.avg_pool3d(x, 3, 1, 1)
    mu_y = F.avg_pool3d(y, 3, 1, 1)
    sig_x = F.avg_pool3d(x*x,3,1,1) - mu_x**2
    sig_y = F.avg_pool3d(y*y,3,1,1) - mu_y**2
    sig_xy= F.avg_pool3d(x*y,3,1,1) - mu_x*mu_y
    ssim = ((2*mu_x*mu_y + C1)*(2*sig_xy + C2))/((mu_x**2+mu_y**2 + C1)*(sig_x+sig_y + C2)+eps)
    return ssim.clamp(0,1).mean()

def train_forecast(loader, epochs=40, lr=2e-4, device="cpu", multitask_seg=True):
    model = DAE_ConvLSTM_Model(multitask_seg=multitask_seg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dice_loss = DiceCELoss(sigmoid=True)
    for ep in range(epochs):
        model.train()
        L=[]
        for batch in loader:
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            t3 = batch["t3"].to(device)
            img3_hat, seg3_hat = model(t1, t2)
            l_rec = F.l1_loss(img3_hat, t3) + (1.0 - ssim3d(img3_hat, t3))
            if multitask_seg and ("m3" in batch) and (batch["m3"] is not None):
                m3 = (batch["m3"]>0.5).float().to(device)
                l_seg = dice_loss(seg3_hat, m3)
            else:
                l_seg = 0.0*img3_hat.mean()
            loss = l_rec + 0.5*l_seg
            opt.zero_grad(); loss.backward(); opt.step()
            L.append(loss.item())
        print(f"[Forecast][{ep+1}/{epochs}] loss={np.mean(L):.4f}")
    return model

@torch.no_grad()
def predict_t3(model, batch, device="cpu", thr=0.5):
    t1 = batch["t1"].to(device); t2 = batch["t2"].to(device)
    img3_hat, seg3_hat = model(t1, t2)
    img3_hat = img3_hat.clamp(0,1)
    seg_bin = (seg3_hat>thr).float() if seg3_hat is not None else None
    return img3_hat, seg3_hat, seg_bin


# =========================
# Évaluation (rapide)
# =========================
@torch.no_grad()
def eval_seg_dice(pred, gt):
    dice = DiceMetric(include_background=False, reduction="mean")
    return dice(pred, gt).item()

@torch.no_grad()
def psnr(x, y, eps=1e-8):
    mse = F.mse_loss(x, y).item()
    if mse < eps: return 99.9
    return 20*math.log10(1.0) - 10*math.log10(mse)


# =========================
# Main / CLI
# =========================
def parse_args():
    ap = argparse.ArgumentParser(description="Tumor Evolution - Script unique (nested/flat/classic)")
    ap.add_argument("--data_root", type=str, default="data", help="Dossier racine des données")
    ap.add_argument("--out_dir", type=str, default="out", help="Sorties (ckpt, prédictions)")
    ap.add_argument("--spacing", type=float, nargs=3, default=(1.2,1.2,1.2), help="Resampling (mm)")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--cache_rate", type=float, default=0.2, help="0..1 fraction du cache RAM MONAI")
    ap.add_argument("--device", type=str, default=None, help="'cuda' ou 'cpu' (auto si None)")

    # modes de scan
    ap.add_argument("--scan_flat", action="store_true", help="Scanner fichiers à plat")
    ap.add_argument("--scan_nested", action="store_true", help="Scanner arborescence imbriquée <PID>/<DATE>/<file>")
    ap.add_argument("--modality", type=str, default="FLAIR", help="Modalité: FLAIR, T1, T2, ...")
    ap.add_argument("--use_last", action="store_true", help="Prendre les 3 acquisitions les plus récentes")

    # actions
    ap.add_argument("--train_diffusion", action="store_true")
    ap.add_argument("--train_forecast", action="store_true")
    ap.add_argument("--infer", action="store_true")

    # hyper
    ap.add_argument("--diff_epochs", type=int, default=20)
    ap.add_argument("--diff_lr", type=float, default=1e-4)
    ap.add_argument("--diff_timesteps", type=int, default=300)
    ap.add_argument("--forecast_epochs", type=int, default=40)
    ap.add_argument("--forecast_lr", type=float, default=2e-4)
    ap.add_argument("--multitask_seg", action="store_true", help="Ajoute tête de segmentation à la partie 2")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--sample_steps", type=int, default=None, help="Pas DDPM en inférence (Partie 1)")

    # checkpoints
    ap.add_argument("--ckpt_diff", type=str, default="out/diffusion_seg.pt")
    ap.add_argument("--ckpt_forecast", type=str, default="out/forecast.pt")
    return ap.parse_args()

def main():
    args = parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")
    ensure_dir(args.out_dir)

    # Sanity-check mapping (affiche 3 items max)
    try:
        if args.scan_nested:
            preview = build_items_nested(args.data_root, modality=args.modality, use_last=args.use_last)
        elif args.scan_flat:
            preview = build_items_flat(args.data_root, modality=args.modality, use_last=args.use_last)
        else:
            preview = build_items_classic(args.data_root)
        print("Mapping exemples:")
        for it in preview[:3]:
            print({k: it[k] for k in ["t1","t2","t3","m1","m2","m3"]})
    except Exception as e:
        print("Sanity-check build_items:", e)

    # loaders
    train_loader, val_loader = get_loaders(
        args.data_root,
        batch_size=args.batch_size,
        spacing=tuple(args.spacing),
        cache_rate=args.cache_rate,
        scan_flat=args.scan_flat,
        scan_nested=args.scan_nested,
        modality=args.modality,
        use_last=args.use_last
    )

    # Entraînement diffusion
    if args.train_diffusion:
        print("=== Entraînement diffusion (segmentation) ===")
        diff_model = train_diffusion(train_loader, epochs=args.diff_epochs, lr=args.diff_lr,
                                     device=dev, timesteps=args.diff_timesteps)
        torch.save(diff_model.state_dict(), args.ckpt_diff)
        print(f"Diffusion sauvegardé -> {args.ckpt_diff}")

    # Entraînement forecast
    if args.train_forecast:
        print("=== Entraînement DAE+ConvLSTM (prévision) ===")
        forecast_model = train_forecast(train_loader, epochs=args.forecast_epochs, lr=args.forecast_lr,
                                        device=dev, multitask_seg=args.multitask_seg)
        torch.save(forecast_model.state_dict(), args.ckpt_forecast)
        print(f"Forecast sauvegardé -> {args.ckpt_forecast}")

    # Inférence / évaluation
    if args.infer:
        print("=== Inférence / Évaluation (validation set) ===")
        diff_model = None
        if Path(args.ckpt_diff).exists():
            diff_model = DiffusionSeg(timesteps=args.diff_timesteps).to(dev)
            diff_model.load_state_dict(torch.load(args.ckpt_diff, map_location=dev))
            diff_model.eval()

        forecast_model = None
        if Path(args.ckpt_forecast).exists():
            forecast_model = DAE_ConvLSTM_Model(multitask_seg=args.multitask_seg).to(dev)
            forecast_model.load_state_dict(torch.load(args.ckpt_forecast, map_location=dev))
            forecast_model.eval()

        dice_scores = []
        psnrs = []
        for i, batch in enumerate(val_loader):
            ref_path = batch["t1_meta_dict"]["filename_or_obj"][0] if "t1_meta_dict" in batch else preview[0]["ref"]

            if diff_model is not None:
                seg_bin, seg_prob = infer_diffusion(diff_model, batch, device=dev, thr=args.thr, steps=args.sample_steps)
                save_nii_like(ref_path, seg_prob[0], os.path.join(args.out_dir, f"val{i:03d}_t3_maskProb.nii.gz"))
                save_nii_like(ref_path, seg_bin[0],  os.path.join(args.out_dir, f"val{i:03d}_t3_maskBin.nii.gz"))
                if ("m3" in batch) and (batch["m3"] is not None):
                    gt = (batch["m3"]>0.5).float().to(dev)
                    d = eval_seg_dice(seg_bin, gt)
                    dice_scores.append(d)
                    print(f"[Val {i}] Dice (diff) = {d:.4f}")

            if forecast_model is not None:
                img3_hat, seg3_hat, seg3_bin = predict_t3(forecast_model, batch, device=dev, thr=args.thr)
                save_nii_like(ref_path, img3_hat[0], os.path.join(args.out_dir, f"val{i:03d}_t3_pred.nii.gz"))
                if seg3_hat is not None:
                    save_nii_like(ref_path, seg3_hat[0], os.path.join(args.out_dir, f"val{i:03d}_t3_predMaskProb.nii.gz"))
                    save_nii_like(ref_path, seg3_bin[0], os.path.join(args.out_dir, f"val{i:03d}_t3_predMaskBin.nii.gz"))
                if "t3" in batch:
                    t3 = batch["t3"].to(dev)
                    p = psnr(img3_hat, t3)
                    s = ssim3d(img3_hat, t3).item()
                    psnrs.append(p)
                    print(f"[Val {i}] PSNR={p:.2f}dB  SSIM={s:.4f}")

        if dice_scores:
            print(f"Dice moyen (diffusion): {np.mean(dice_scores):.4f}")
        if psnrs:
            print(f"PSNR moyen (forecast): {np.mean(psnrs):.2f} dB")
        print(f"Sorties NIfTI -> {args.out_dir}")

if __name__ == "__main__":
    main()
