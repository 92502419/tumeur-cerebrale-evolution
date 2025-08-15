import os
import io
import math
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ====== Import des modèles depuis main.py si dispo, sinon fallback minimal ======
# On essaie d'importer la définition exacte du projet pour recharger les poids.
DAE_IMPORTED = False
DIFF_IMPORTED = False
try:
    from main import DAE_ConvLSTM_Model as DAE_ConvLSTM_Model_Proj  # type: ignore
    DAE_IMPORTED = True
except Exception:
    pass

try:
    from main import DiffusionSeg as DiffusionSeg_Proj  # type: ignore
    DIFF_IMPORTED = True
except Exception:
    pass


# ====== Fallbacks (si main.py non importable) ======
class Encoder3D(nn.Module):
    def __init__(self, in_ch=1, base=32, z_ch=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv3d(in_ch, base, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base, base, 3, stride=2, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base, base*2, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base*2, base*2, 3, stride=2, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base*2, z_ch, 3, padding=1), nn.LeakyReLU(inplace=True),
        )
    def forward(self, x): return self.enc(x)

class Decoder3D(nn.Module):
    def __init__(self, out_ch=1, base=32, z_ch=128):
        super().__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(z_ch, base*2, 2, stride=2), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base*2, base*2, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(base*2, base, 2, stride=2), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base, base, 3, padding=1), nn.LeakyReLU(inplace=True),
            nn.Conv3d(base, out_ch, 1),
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
        h = [None]*len(self.layers); c = [None]*len(self.layers)
        out = None
        for x in xs:
            for i,cell in enumerate(self.layers):
                h[i], c[i] = cell(x if i==0 else h[i-1], h[i], c[i])
            out = h[-1]
        return out

class DAE_ConvLSTM_Model_Fallback(nn.Module):
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

# Diffusion fallback simple (sampling lent mais fonctionnel)
class EpsilonUNet3D(nn.Module):
    def __init__(self, in_ch=2, out_ch=1):
        super().__init__()
        # un petit U-Net à la main pour éviter MONAI ici
        ch = 16
        self.d1 = nn.Sequential(nn.Conv3d(in_ch, ch, 3, padding=1), nn.ReLU())
        self.d2 = nn.Sequential(nn.Conv3d(ch, ch*2, 3, stride=2, padding=1), nn.ReLU())
        self.d3 = nn.Sequential(nn.Conv3d(ch*2, ch*4, 3, stride=2, padding=1), nn.ReLU())
        self.u2 = nn.Sequential(nn.ConvTranspose3d(ch*4, ch*2, 2, stride=2), nn.ReLU())
        self.u1 = nn.Sequential(nn.ConvTranspose3d(ch*2, ch, 2, stride=2), nn.ReLU())
        self.out = nn.Conv3d(ch, out_ch, 1)
    def forward(self, x):
        x1 = self.d1(x); x2 = self.d2(x1); x3 = self.d3(x2)
        u2 = self.u2(x3); u1 = self.u1(u2)
        return self.out(u1)

class DiffusionSeg_Fallback(nn.Module):
    def __init__(self, timesteps=100):
        super().__init__()
        self.T = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.eps_net = EpsilonUNet3D()
    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        a_bar = self.alphas_cumprod[t].view(-1,1,1,1,1)
        return a_bar.sqrt()*x0 + (1-a_bar).sqrt()*noise
    def p_losses(self, img, x0, t):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps = self.eps_net(torch.cat([img, x_t], dim=1))
        return F.mse_loss(eps, noise)
    @torch.no_grad()
    def sample(self, img, shape, steps=None, device="cpu"):
        T = self.T if steps is None else min(steps, self.T)
        x = torch.randn(shape, device=device)
        for t in reversed(range(T)):
            eps = self.eps_net(torch.cat([img, x], dim=1))
            beta_t = self.betas[t].to(device)
            alpha_t = 1.0 - beta_t
            a_bar_t = self.alphas_cumprod[t].to(device)
            x = (x - ((1 - alpha_t)/((1 - a_bar_t).sqrt()))*eps) / (alpha_t**0.5)
            if t > 0:
                x = x + (beta_t**0.5)*torch.randn_like(x)
        return torch.sigmoid(x)

# ====== Utils ======
def device_auto() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _tmp_copy(upload) -> str:
    """Sauvegarde un UploadedFile dans un tmp et retourne le chemin."""
    suffix = ".nii.gz" if upload.name.endswith(".gz") else ".nii"
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tf.write(upload.getbuffer())
    tf.flush(); tf.close()
    return tf.name

def load_nifti_any(src) -> Tuple[np.ndarray, dict]:
    """Charge un NIfTI depuis path ou UploadedFile; retourne (vol, meta)."""
    if hasattr(src, "read"):  # UploadedFile streamlit
        src = _tmp_copy(src)
    img = nib.load(str(src))
    arr = img.get_fdata().astype(np.float32)
    meta = {"affine": img.affine, "header": img.header, "zooms": img.header.get_zooms()}
    return arr, meta

def norm01_percentile(x: np.ndarray, pmin=1.0, pmax=99.0) -> np.ndarray:
    a = np.percentile(x, pmin)
    b = np.percentile(x, pmax)
    if b - a < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1).astype(np.float32)

def center_crop_or_pad(x: np.ndarray, target: Tuple[int,int,int]) -> np.ndarray:
    """Centre-crop ou pad en 3D à la taille target (D,H,W)."""
    D,H,W = x.shape
    Td,Th,Tw = target
    out = np.zeros((Td,Th,Tw), dtype=x.dtype)
    sd = max(0, (D - Td)//2); sh = max(0, (H - Th)//2); sw = max(0, (W - Tw)//2)
    td = max(0, (Td - D)//2); th = max(0, (Th - H)//2); tw = max(0, (Tw - W)//2)
    cd = min(D, Td); ch = min(H, Th); cw = min(W, Tw)
    out[td:td+cd, th:th+ch, tw:tw+cw] = x[sd:sd+cd, sh:sh+ch, sw:sw+cw]
    return out

def make_multiple_of(x: Tuple[int,int,int], k=4) -> Tuple[int,int,int]:
    return tuple(int(math.ceil(s / k) * k) for s in x)

def stats_df(x: np.ndarray) -> pd.DataFrame:
    q = np.percentile(x, [0.5,1,5,50,95,99,99.5])
    dat = {
        "min":[float(x.min())], "max":[float(x.max())], "mean":[float(x.mean())], "std":[float(x.std())],
        "q0.5":[q[0]], "q1":[q[1]], "q5":[q[2]], "q50":[q[3]], "q95":[q[4]], "q99":[q[5]], "q99.5":[q[6]],
        "shape":[str(x.shape)]
    }
    return pd.DataFrame(dat)

def plot_slice(img2d: np.ndarray, title: str = "", cmap="gray"):
    # figure carrée pour éviter tout étirement
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(img2d, cmap=cmap)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.axis("off")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

def overlay_mask(img2d: np.ndarray, mask2d: np.ndarray, alpha: float = 0.4):
    img = (img2d - img2d.min()) / max(1e-6, img2d.max() - img2d.min())
    rgb = np.stack([img, img, img], axis=-1)
    red = np.zeros_like(rgb); red[..., 0] = 1.0
    out = (1 - alpha) * rgb + alpha * red * (mask2d > 0)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(out)
    ax.set_title("Overlay masque")
    ax.set_aspect("equal")
    ax.axis("off")
    st.pyplot(fig, use_column_width=False)
    plt.close(fig)


def ssim3d(x: torch.Tensor, y: torch.Tensor, C1=0.01**2, C2=0.03**2, eps=1e-6) -> torch.Tensor:
    mu_x = F.avg_pool3d(x, 3, 1, 1)
    mu_y = F.avg_pool3d(y, 3, 1, 1)
    sig_x = F.avg_pool3d(x*x,3,1,1) - mu_x**2
    sig_y = F.avg_pool3d(y*y,3,1,1) - mu_y**2
    sig_xy= F.avg_pool3d(x*y,3,1,1) - mu_x*mu_y
    ssim = ((2*mu_x*mu_y + C1)*(2*sig_xy + C2))/((mu_x**2+mu_y**2 + C1)*(sig_x+sig_y + C2)+eps)
    return ssim.clamp(0,1).mean()

# ====== Cache ======
@st.cache_resource
def load_forecast_model(ckpt_path: str, device: str = "cpu", multitask_seg: bool = True):
    if not Path(ckpt_path).exists(): return None
    Model = DAE_ConvLSTM_Model_Proj if DAE_IMPORTED else DAE_ConvLSTM_Model_Fallback
    model = Model(multitask_seg=multitask_seg).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model

@st.cache_resource
def load_diffusion_model(ckpt_path: str, device: str = "cpu", timesteps: int = 300):
    if not Path(ckpt_path).exists(): return None
    Model = DiffusionSeg_Proj if DIFF_IMPORTED else DiffusionSeg_Fallback
    model = Model(timesteps=timesteps).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model

# ====== UI ======
st.set_page_config(page_title="Tumor Evolution Dashboard", layout="centered")
st.title(" EVOLUTION DE TUMEUR — TABLEAU DE BORD")

with st.sidebar:
    st.header("Importation & Options")
    mode = st.radio("Mode d'analyse", ["Image unique", "Série temporelle (t1,t2,[t3])"], index=1)
    uploads = st.file_uploader("Importer NIfTI (.nii/.nii.gz)", type=["nii","nii.gz"], accept_multiple_files=True)
    spacing_mm = st.slider("Downsample (affichage seulement) — Pas (mm)", 0.5, 3.0, 1.0, 0.1)
    do_norm = st.checkbox("Normaliser (percentiles 1–99 → [0,1])", True)
    do_prep = st.checkbox("Prétraiter pour AE (crop/pad multiples de 4)", True)
    run_btn = st.button("🔎 Analyser maintenant")

    st.divider()
    st.subheader("Modèles & Logs")
    dev = device_auto()
    st.caption(f"Device: **{dev}**")
    ckpt_forecast = st.text_input("Checkpoint Forecast (AutoEncoder+ConvLSTM)", "out/forecast.pt")
    ckpt_diff = st.text_input("Checkpoint Diffusion (Segmentation)", "out/diffusion_seg.pt")
    sample_steps = st.slider("Diffusion — steps (inférence)", 10, 300, 60, 10)
    st.caption(" Plus de steps = plus lent, potentiellement plus précis.")

    st.divider()
    log_forecast = st.text_input("Log perte Forecast (CSV)", "out/forecast_log.csv")
    log_diff = st.text_input("Log perte Diffusion (CSV)", "out/diffusion_log.csv")

# ====== Corps principal ======

def maybe_downsample_for_display(vol: np.ndarray, zooms: Tuple[float,float,float], step: float = 1.0) -> Tuple[np.ndarray, Tuple[float,float,float]]:
    """
    Sous-échantillonne isotropiquement pour l'affichage (rapide) en respectant la physique.
    step=1 => pas de downsample ; step≈2 => prend 1 voxel sur 2 dans chaque axe.
    Retourne: (vol_ds, zooms_ds) où zooms sont multipliés par le facteur de stride.
    """
    s = max(1, int(round(step)))
    if s == 1:
        return vol, zooms
    vol_ds = vol[::s, ::s, ::s]
    zooms_ds = (zooms[0]*s, zooms[1]*s, zooms[2]*s)  # (X, Y, Z) en mm
    return vol_ds, zooms_ds


def show_basic_views(vol: np.ndarray, zooms_xyz: Tuple[float,float,float], name="image"):
    # vol shape = (X, Y, Z) avec nibabel ; zooms = (dx, dy, dz) en mm
    X, Y, Z = vol.shape
    st.write(f"**{name}** — shape: `{vol.shape}`  |  zooms (mm): {tuple(round(z,3) for z in zooms_xyz)}")

    with st.expander(f" Visualisation 2D — {name}", expanded=True):
        axis = st.radio(f"Axe (pour {name})", ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"], horizontal=True, key=f"axis_{name}")

        if axis == "Axial (Z)":
            idx = st.slider(f"Indice de coupe Z (0..{Z-1})", 0, Z-1, Z//2, key=f"idx_{name}_z")
            img2d = vol[:, :, idx]          # (X,Y)
            # lignes = Y, colonnes = X
            pix_row_mm, pix_col_mm = zooms_xyz[1], zooms_xyz[0]
            plot_slice_phys(img2d.T, pix_row_mm, pix_col_mm, title=f"{name} — Axial (Z)")  # transpose pour afficher (Y,X)

            # heatmap coupe centrale (Z//2)
            c2d = vol[:, :, Z//2]
            plot_slice_phys(c2d.T, zooms_xyz[1], zooms_xyz[0], title=f"{name} — Heatmap centrale (Axial)", cmap="hot")

        elif axis == "Coronal (Y)":
            idx = st.slider(f"Indice de coupe Y (0..{Y-1})", 0, Y-1, Y//2, key=f"idx_{name}_y")
            img2d = vol[:, idx, :]          # (X,Z)
            # lignes = Z, colonnes = X
            pix_row_mm, pix_col_mm = zooms_xyz[2], zooms_xyz[0]
            plot_slice_phys(img2d.T, pix_row_mm, pix_col_mm, title=f"{name} — Coronal (Y)")

            c2d = vol[:, Y//2, :]
            plot_slice_phys(c2d.T, zooms_xyz[2], zooms_xyz[0], title=f"{name} — Heatmap centrale (Coronal)", cmap="hot")

        else:  # Sagittal (X)
            idx = st.slider(f"Indice de coupe X (0..{X-1})", 0, X-1, X//2, key=f"idx_{name}_x")
            img2d = vol[idx, :, :]          # (Y,Z)
            # lignes = Z, colonnes = Y
            pix_row_mm, pix_col_mm = zooms_xyz[2], zooms_xyz[1]
            plot_slice_phys(img2d, pix_row_mm, pix_col_mm, title=f"{name} — Sagittal (X)")

            c2d = vol[X//2, :, :]
            plot_slice_phys(c2d, zooms_xyz[2], zooms_xyz[1], title=f"{name} — Heatmap centrale (Sagittal)", cmap="hot")


def run_forecast_inference(model, t1: np.ndarray, t2: np.ndarray, t3: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Retourne (t3_pred, seg_prob, err_map) en numpy."""
    device = next(model.parameters()).device
    def to_tensor(x):
        x = torch.from_numpy(x[None, None]).to(device)
        return x
    T1 = to_tensor(t1); T2 = to_tensor(t2)
    with torch.no_grad():
        img3_hat, seg3_hat = model(T1, T2)
        img3_hat = img3_hat.clamp(0,1)
        pred = img3_hat[0,0].cpu().numpy()
        seg_prob = seg3_hat[0,0].cpu().numpy() if seg3_hat is not None else None
    err = None
    if t3 is not None:
        err = np.abs(pred - t3)
    return pred, seg_prob, err

def run_diffusion_inference(model, img: np.ndarray, steps: int) -> np.ndarray:
    device = next(model.parameters()).device
    X = torch.from_numpy(img[None, None]).to(device)
    with torch.no_grad():
        prob = model.sample(X, shape=X.shape, steps=steps, device=device)
        prob = prob[0,0].clamp(0,1).cpu().numpy()
    return prob

def load_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists(): return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None

def plot_slice_phys(img2d: np.ndarray, pix_row_mm: float, pix_col_mm: float, title: str = "", cmap="gray"):
    """
    Affiche une coupe 2D en respectant la taille physique des pixels:
    - 'pix_row_mm' = taille (mm) d'un pixel vertical (axe des lignes)
    - 'pix_col_mm' = taille (mm) d'un pixel horizontal (axe des colonnes)
    """
    h, w = img2d.shape
    extent = [0, w*pix_col_mm, 0, h*pix_row_mm]  # x_min, x_max, y_min, y_max en mm
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    im = ax.imshow(img2d, cmap=cmap, origin="lower", extent=extent, aspect="equal", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

def overlay_mask_phys(img2d: np.ndarray, mask2d: np.ndarray, pix_row_mm: float, pix_col_mm: float, alpha: float = 0.35):
    """Superpose un masque en respectant la physique (mm)."""
    h, w = img2d.shape
    extent = [0, w*pix_col_mm, 0, h*pix_row_mm]
    img = (img2d - img2d.min()) / max(1e-6, img2d.max()-img2d.min())

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.imshow(img, cmap="gray", origin="lower", extent=extent, aspect="equal", interpolation="nearest")
    ax.imshow(mask2d>0, cmap="Reds", origin="lower", extent=extent, aspect="equal", alpha=alpha, interpolation="nearest")
    ax.set_title("Overlay masque (échelle mm)")
    ax.set_xlabel("mm"); ax.set_ylabel("mm")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

# ====== Actions ======
if run_btn and uploads:
    # tri par nom pour stabilité
    uploads_sorted = sorted(uploads, key=lambda u: u.name)

    # Charger 1..3 volumes
    vols = []
    metas = []
    for up in uploads_sorted[:3]:
        vol, meta = load_nifti_any(up)
        if do_norm:
            vol = norm01_percentile(vol)
        vol_disp = maybe_downsample_for_display(vol, spacing_mm)
        vols.append(vol_disp)
        metas.append(meta)

    # === Section 1 : Visualisation & Statistiques (UNE COLONNE) ===
    st.subheader(" VISUALISATIONS ET STATISTIQUES")

    tabs = st.tabs([f"Image {i+1}" for i in range(len(vols))])
    for i, tab in enumerate(tabs):
        with tab:
            show_basic_views(vols[i], name=f"image{i+1}")
            df = stats_df(vols[i])
            st.dataframe(df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 3), dpi=120)
            ax.hist(vols[i].ravel(), bins=64)
            ax.set_title("Histogramme des intensités")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

    # === Section 2 : Prétraitement AE ===
    if do_prep:
        st.subheader(" Prétraitement pour AutoEncoder 3D")
        shapes = np.array([v.shape for v in vols])
        med = np.median(shapes, axis=0).astype(int)
        target = make_multiple_of(tuple(med.tolist()), 4)
        st.caption(f"Taille cible (multiple de 4) : {target}")
        prepped = [center_crop_or_pad(v, target) for v in vols]
        for i, v in enumerate(prepped):
            st.write(f"image{i+1} → {v.shape}")
        vols = prepped

    # === Section 3 : Interprétations automatiques ===
    st.subheader(" Interprétations")

    # Reconstruction & anomalies (Forecast)
    model_forecast = load_forecast_model(ckpt_forecast, device=device_auto(), multitask_seg=True)
    if model_forecast is not None:
        st.markdown("**Reconstruction & Anomalies (AE+ConvLSTM)**")
        if mode.startswith("Série") and len(vols) >= 2:
            t1, t2 = vols[0], vols[1]
            t3_true = vols[2] if len(vols) >= 3 else None
            pred, seg_prob, err = run_forecast_inference(model_forecast, t1, t2, t3_true)

            st.caption("Prédiction t3 (reconstruite/évolutive)")
            plot_slice(pred[pred.shape[0]//2, :, :], "t3_pred — coupe axiale")

            if t3_true is not None:
                st.caption("Erreur absolue |t3_pred - t3| (anomalies)")
                plot_slice(err[err.shape[0]//2, :, :], "Erreur — heatmap", cmap="hot")

            if seg_prob is not None:
                st.caption("Segmentation (probabilité) issue du latent")
                plot_slice(seg_prob[seg_prob.shape[0]//2, :, :], "Seg prob — heatmap", cmap="hot")
        else:
            st.info("Pour la reconstruction/évolution, charge au moins **2 images** (t1 & t2).")
    else:
        st.warning("Aucun checkpoint Forecast trouvé. Place **out/forecast.pt** pour activer la reconstruction.")

    # Segmentation par diffusion
    model_diff = load_diffusion_model(ckpt_diff, device=device_auto(), timesteps=max(100, sample_steps))
    if model_diff is not None:
        st.markdown("**Segmentation (Diffusion)**")
        img_seg = vols[2] if (len(vols) >= 3 and mode.startswith("Série")) else vols[0]
        prob = run_diffusion_inference(model_diff, img_seg, steps=sample_steps)
        thr = st.slider("Seuil binaire", 0.05, 0.95, 0.5, 0.05)
        mask = (prob > thr).astype(np.float32)
        st.caption("Probabilité de masque (coupe centrale)")
        plot_slice(prob[prob.shape[0]//2, :, :], "Mask prob — heatmap", cmap="hot")
        st.caption("Overlay masque / image")
        overlay_mask(img_seg[img_seg.shape[0]//2, :, :], mask[mask.shape[0]//2, :, :], alpha=0.35)
    else:
        st.info("Aucun checkpoint Diffusion trouvé (optionnel). Place **out/diffusion_seg.pt** pour activer la segmentation.")

    # === Section 4 : Évolution de la perte ===
    st.subheader(" Évolution de la perte")
    df_f = load_csv_if_exists(log_forecast)
    df_d = load_csv_if_exists(log_diff)
    if (df_f is None) and (df_d is None):
        st.caption("Aucun log CSV trouvé. (Option : écrivez un CSV des pertes pendant l'entraînement.)")
    if df_f is not None:
        st.write("Forecast")
        st.line_chart(df_f.set_index(df_f.columns[0]))
    if df_d is not None:
        st.write("Diffusion")
        st.line_chart(df_d.set_index(df_d.columns[0]))

else:
    st.info(" Importez 1 à 3 fichiers NIfTI dans la barre latérale, ajustez les options, puis cliquez sur **Analyser maintenant**.")
    st.caption("Astuce : pour l'évolution spatio-temporelle, chargez t1, t2 (et optionnellement t3 pour comparer).")
