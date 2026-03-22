# ================================================================
# Train with live curves + periodic full-val prediction & plots
# Supports batches shaped as (xb, yb) OR (xb, pb, yb) OR dicts.
#
# UPDATED:
# - CUDA AMP support (autocast + GradScaler)
# - Saves:
#     best checkpoint:  <save_dir>/best.ckpt
#     last checkpoint:  <save_dir>/last.ckpt   (every epoch)
# - Proper resume: restores model/opt/sched/scaler + epoch
# - Still writes learning_curves.csv + loss PNGs + periodic scatter/metrics
# ================================================================

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from sklearn.metrics import r2_score


# ---------------------- Helpers ----------------------

def _unpack_batch(batch):
    """
    Accept:
      - (xb, yb)
      - (xb, pb, yb)
      - dict with possible keys: x/input, y/target, phase/phi/p
    Return: xb, pb_or_None, yb
    """
    if isinstance(batch, dict):
        xb = batch.get("x") or batch.get("X") or batch.get("input")
        yb = batch.get("y") or batch.get("Y") or batch.get("target")
        pb = batch.get("phase") or batch.get("phi") or batch.get("p")
        return xb, pb, yb

    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            xb, yb = batch
            return xb, None, yb
        if len(batch) == 3:
            xb, pb, yb = batch
            return xb, pb, yb

    raise ValueError("Batch must be (xb, yb), (xb, pb, yb), or a dict with x/y/(phase).")


def _sse_per_target(pred, y):
    """
    Works for output shapes:
      - (B, D)
      - (B, T, D)
      - (B, ..., D)
    Returns CPU Tensor of shape (D,)
    """
    se = (pred - y) ** 2
    for _ in range(se.dim() - 1):
        se = se.sum(dim=0)
    return se.detach().to("cpu")


def _forward_maybe_phase(model, xb, pb):
    return model(xb, pb) if pb is not None else model(xb)


def _destandardize(y, y_mean=None, y_std=None):
    if y_mean is None or y_std is None:
        return y
    y_mean_np = y_mean.squeeze().detach().cpu().numpy() if torch.is_tensor(y_mean) else np.asarray(y_mean)
    y_std_np  = y_std.squeeze().detach().cpu().numpy()  if torch.is_tensor(y_std)  else np.asarray(y_std)
    return y * y_std_np + y_mean_np


def _predict_on_val(model, val_loader, device):
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for batch in val_loader:
            xb, pb, yb = _unpack_batch(batch)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True) if pb is not None else None
            pred = _forward_maybe_phase(model, xb, pb)
            yp.append(pred.detach().cpu())
            yt.append(yb.detach().cpu())
    y_true = torch.cat(yt, dim=0).numpy()
    y_pred = torch.cat(yp, dim=0).numpy()
    return y_true, y_pred


def _plot_true_vs_pred_multi(y_true, y_pred, idxs=(0, 1, 2), target_names=None, out_png=None):
    cols = len(idxs)
    plt.figure(figsize=(4 * cols, 4))
    for j, i in enumerate(idxs):
        ax = plt.subplot(1, cols, j + 1)
        ax.scatter(y_true[:, i], y_pred[:, i], s=10, alpha=0.6)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], lw=1, color="k")
        name = (target_names[i] if (target_names and i < len(target_names)) else f"param {i}")
        ax.set_title(f"{name}: true vs pred")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.grid(ls="--", alpha=0.4)
        ax.axis("equal")
    plt.tight_layout()
    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=160)
    plt.close()


def _write_metrics_csv(path, epoch, names, mse, mae, r2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "target", "MSE", "MAE", "R2"])
        for i in range(len(mse)):
            name = (names[i] if (names and i < len(names)) else f"param_{i}")
            w.writerow([epoch, name, float(mse[i]), float(mae[i]), float(r2[i])])


def _save_ckpt(path, *, epoch, model, opt, sched, scaler, best_val, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "opt": opt.state_dict() if opt is not None else None,
        "sched": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val": float(best_val) if best_val is not None else None,
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def _load_ckpt(path, model, opt=None, sched=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if opt is not None and ckpt.get("opt") is not None:
        opt.load_state_dict(ckpt["opt"])
    if sched is not None and ckpt.get("sched") is not None:
        sched.load_state_dict(ckpt["sched"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = int(ckpt.get("epoch", 0))
    best_val = ckpt.get("best_val", None)
    return epoch, best_val, ckpt.get("extra", {})


# ---------------------- Main Trainer ----------------------

def train_with_display_and_save(
    model, train_loader, val_loader,
    *,
    epochs=100,
    lr=3e-4,
    wd=1e-4,
    clip=1.0,
    t_max=120,
    device=None,
    target_names=None,
    save_dir="epoch_plots",
    print_every=1,
    display_epochs=None,      # int interval or list[int]
    save_epochs=None,         # int interval or list[int]
    eval_every=5,
    pred_idxs=(0, 1, 2),
    y_mean=None,
    y_std=None,
    use_amp=None,             # None -> auto on cuda
    resume=True,              # resume from <save_dir>/last.ckpt if exists
):
    """
    Batches can be:
      (xb, yb)  OR  (xb, pb, yb)  OR  {"x":..., "y":..., "phase":...}
    """

    os.makedirs(save_dir, exist_ok=True)
    csv_path   = os.path.join(save_dir, "learning_curves.csv")
    best_path  = os.path.join(save_dir, "best.ckpt")
    last_path  = os.path.join(save_dir, "last.ckpt")

    device = device or next(model.parameters()).device
    model.to(device)

    # AMP
    if use_amp is None:
        use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp))

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
    lossf = nn.MSELoss(reduction="mean")

    # histories (resume from CSV for plotting continuity)
    tr_hist, va_hist = [], []
    last_epoch_from_csv = 0
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                _ = next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    ep_i = int(row[0])
                    tr_hist.append(float(row[1]))
                    va_hist.append(float(row[2]))
                    last_epoch_from_csv = ep_i
        except Exception:
            tr_hist, va_hist, last_epoch_from_csv = [], [], 0

    # init CSV if needed
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_mse", "val_mse"])

    # resume from last checkpoint (authoritative for epoch/weights/opt/sched/scaler)
    start_epoch = 1
    best_val = float("inf")

    if resume and os.path.exists(last_path):
        ep0, best0, _ = _load_ckpt(last_path, model, opt=opt, sched=sched, scaler=scaler, device=device)
        start_epoch = ep0 + 1
        if best0 is not None:
            best_val = float(best0)

    # curve figure
    fig, ax = plt.subplots(figsize=(7, 4))
    (ltr,) = ax.plot(range(1, len(tr_hist) + 1), tr_hist, label="Train MSE")
    (lva,) = ax.plot(range(1, len(va_hist) + 1), va_hist, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()

    per_target_train_hist = []
    per_target_val_hist   = []

    def _should(ep, control):
        if control is None:
            return False
        if isinstance(control, int):
            return (ep % control == 0) or (ep == epochs)
        if isinstance(control, (list, tuple, set)):
            return ep in control
        return False

    # main loop
    for ep in range(start_epoch, epochs + 1):
        # ---------------- train ----------------
        model.train()
        train_loss, n_samples = 0.0, 0
        per_target_sse = None

        for batch in train_loader:
            xb, pb, yb = _unpack_batch(batch)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True) if pb is not None else None

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred = _forward_maybe_phase(model, xb, pb)
                    loss = lossf(pred, yb)
                scaler.scale(loss).backward()
                if clip is not None and clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(opt)
                scaler.update()
            else:
                pred = _forward_maybe_phase(model, xb, pb)
                loss = lossf(pred, yb)
                loss.backward()
                if clip is not None and clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()

            bs = xb.size(0)
            n_samples += bs
            train_loss += loss.item() * bs

            sse_batch = _sse_per_target(pred, yb)
            per_target_sse = sse_batch if per_target_sse is None else (per_target_sse + sse_batch)

        train_loss /= max(n_samples, 1)
        tr_hist.append(train_loss)
        tr_vec = (per_target_sse / max(n_samples, 1)).detach().cpu().numpy()
        per_target_train_hist.append(tr_vec)

        # ---------------- val ----------------
        model.eval()
        val_loss, n_samples = 0.0, 0
        per_target_sse = None

        with torch.no_grad():
            for batch in val_loader:
                xb, pb, yb = _unpack_batch(batch)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pb = pb.to(device, non_blocking=True) if pb is not None else None

                pred = _forward_maybe_phase(model, xb, pb)
                loss = lossf(pred, yb)

                bs = xb.size(0)
                n_samples += bs
                val_loss += loss.item() * bs

                sse_batch = _sse_per_target(pred, yb)
                per_target_sse = sse_batch if per_target_sse is None else (per_target_sse + sse_batch)

        val_loss /= max(n_samples, 1)
        va_hist.append(val_loss)
        va_vec = (per_target_sse / max(n_samples, 1)).detach().cpu().numpy()
        per_target_val_hist.append(va_vec)

        sched.step()

        # ---------------- save last + best ----------------
        _save_ckpt(
            last_path,
            epoch=ep,
            model=model,
            opt=opt,
            sched=sched,
            scaler=scaler,
            best_val=best_val,
        )

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            _save_ckpt(
                best_path,
                epoch=ep,
                model=model,
                opt=opt,
                sched=sched,
                scaler=scaler,
                best_val=best_val,
            )

        # ---------------- log csv ----------------
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([ep, train_loss, val_loss])

        # ---------------- title string ----------------
        names = target_names if target_names else [f"t{i}" for i in range(len(tr_vec))]
        per_tr = "  ".join([f"{names[i]}={tr_vec[i]:.6g}" for i in range(len(tr_vec))])
        per_va = "  ".join([f"{names[i]}={va_vec[i]:.6g}" for i in range(len(va_vec))])
        lr_now = opt.param_groups[0]["lr"]

        wp_list = []
        if hasattr(model, "w_pos"):
            try: wp_list.append(f"w_pos={float(model.w_pos.detach().cpu()):.4f}")
            except Exception: pass
        if hasattr(model, "w_phase"):
            try: wp_list.append(f"w_phase={float(model.w_phase.detach().cpu()):.4f}")
            except Exception: pass
        if hasattr(model, "w_snr"):
            try: wp_list.append(f"w_snr={float(model.w_snr.detach().cpu()):.4f}")
            except Exception: pass

        wp_txt = (" | " + ", ".join(wp_list)) if wp_list else ""

        title_text = (
            f"Epoch {ep:03d} | Train={train_loss:.6e} | Val={val_loss:.6e} | LR={lr_now:.2e}{wp_txt}\n"
            f"Train per-target: {per_tr}\n"
            f"Val   per-target: {per_va}"
        )

        # ---------------- display/save loss curves ----------------
        if _should(ep, display_epochs) or _should(ep, save_epochs):
            xs = range(1, len(tr_hist) + 1)
            ltr.set_data(xs, tr_hist)
            lva.set_data(xs, va_hist)
            ax.relim()
            ax.autoscale_view()
            ax.set_title(title_text, fontsize=8)
            fig.tight_layout()

            if _should(ep, display_epochs):
                clear_output(wait=True)
                display(fig)
                fig.canvas.draw()
                fig.canvas.flush_events()

            if _should(ep, save_epochs):
                out_total = os.path.join(save_dir, "curve_total_latest.png")
                fig.savefig(out_total, dpi=160)

                # per-target curve
                try:
                    arr_tr = np.vstack(per_target_train_hist)
                    arr_va = np.vstack(per_target_val_hist)
                    n_epochs, n_targets = arr_tr.shape
                    plt.figure(figsize=(7, 4 + 1.0 * (n_targets > 3)))
                    xs_arr = np.arange(1, n_epochs + 1)
                    for j in range(n_targets):
                        nm = names[j] if j < len(names) else f"t{j}"
                        plt.plot(xs_arr, arr_tr[:, j], label=f"Train {nm}")
                        plt.plot(xs_arr, arr_va[:, j], linestyle="--", label=f"Val {nm}")
                    plt.xlabel("Epoch")
                    plt.ylabel("MSE")
                    plt.title("Per-target Train/Val MSE")
                    plt.legend(fontsize=8, ncol=2)
                    plt.grid(ls="--", alpha=0.3)
                    plt.tight_layout()
                    out_per = os.path.join(save_dir, "curve_per_target_latest.png")
                    plt.savefig(out_per, dpi=160)
                    plt.close()
                except Exception:
                    pass

        if ep % print_every == 0:
            print(title_text)

        # ---------------- periodic evaluation ----------------
        if eval_every and (ep % eval_every == 0 or ep == epochs):
            try:
                y_true, y_pred = _predict_on_val(model, val_loader, device)
                y_true_phys = _destandardize(y_true, y_mean, y_std)
                y_pred_phys = _destandardize(y_pred, y_mean, y_std)

                mse = ((y_true_phys - y_pred_phys) ** 2).mean(axis=0)
                mae = np.abs(y_true_phys - y_pred_phys).mean(axis=0)
                r2  = np.array([r2_score(y_true_phys[:, i], y_pred_phys[:, i])
                                for i in range(y_true_phys.shape[1])])

                print("\n--- Periodic Evaluation @ epoch", ep, "---")
                for i in range(len(mse)):
                    nm = names[i] if i < len(names) else f"param_{i}"
                    print(f"  {nm:>12s} | MSE={mse[i]:.6e} | MAE={mae[i]:.6e} | R2={r2[i]:.4f}")

                metrics_csv = os.path.join(save_dir, f"metrics_epoch_{ep:04d}.csv")
                _write_metrics_csv(metrics_csv, ep, names, mse, mae, r2)

                scatter_png = os.path.join(save_dir, f"pred_scatter_epoch_{ep:04d}.png")
                idxs = tuple([i for i in pred_idxs if i < y_true_phys.shape[1]]) or (0,)
                _plot_true_vs_pred_multi(y_true_phys, y_pred_phys, idxs=idxs,
                                         target_names=target_names, out_png=scatter_png)
            except Exception as e:
                print(f"⚠️ Periodic evaluation failed at epoch {ep}: {e}")

    plt.close(fig)

    # restore best at end
    if os.path.exists(best_path):
        _load_ckpt(best_path, model, opt=None, sched=None, scaler=None, device=device)
        model.to(device)
        model.eval()

    curves = {
        "train_total": tr_hist,
        "val_total": va_hist,
        "best_val": float(best_val),
        "best_ckpt": best_path,
        "last_ckpt": last_path,
        "csv_path": csv_path,
        "last_epoch": len(tr_hist),
    }
    return model, curves

