# ================================================================
# Train with live curves + periodic full-val prediction & plots
# Supports batches shaped as (xb, yb) OR (xb, pb, yb) OR dicts.
# Saves:
#   - best weights -> {save_dir}/best_weights.pt (or custom save_path)
#   - learning curves CSV -> {save_dir}/learning_curves.csv
#   - curve PNGs -> {save_dir}/curve_total_latest.png (updated)
#   - per-target curve PNGs -> {save_dir}/curve_per_target_latest.png (updated)
#   - scatter plots (true vs pred) -> {save_dir}/pred_scatter_epoch_####.png
#   - metrics CSV (per eval) -> {save_dir}/metrics_epoch_####.csv
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
        elif len(batch) == 3:
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
    """
    y: np.ndarray [N, D]
    y_mean/y_std: torch.Tensor or np.ndarray shaped [D] (or broadcastable)
    """
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

def _plot_true_vs_pred_multi(y_true, y_pred, idxs=(0,1,2), target_names=None, out_png=None):
    cols = len(idxs)
    plt.figure(figsize=(4*cols, 4))
    for j, i in enumerate(idxs):
        ax = plt.subplot(1, cols, j+1)
        ax.scatter(y_true[:, i], y_pred[:, i], s=10, alpha=0.6)
        lo = min(y_true[:, i].min(), y_pred[:, i].min())
        hi = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], lw=1, color="k")
        name = (target_names[i] if (target_names and i < len(target_names)) else f"param {i}")
        ax.set_title(f"{name}: true vs pred")
        ax.set_xlabel("True"); ax.set_ylabel("Pred")
        ax.grid(ls="--", alpha=0.4); ax.axis("equal")
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

# ---------------------- Main Trainer ----------------------

def train_with_display_and_save(
    model, train_loader, val_loader,
    *,
    epochs=100,                 # total epochs desired (global), not "additional"
    lr=3e-4,
    wd=1e-4,
    clip=1.0,
    t_max=120,
    save_path=None,            # default -> {save_dir}/best_weights.pt
    device=None,
    target_names=None,
    save_dir="epoch_plots",
    print_every=1,
    display_epochs=None,       # list[int] or int interval
    save_epochs=None,          # list[int] or int interval
    # ------- NEW knobs for periodic evaluation -------
    eval_every=5,              # run full val prediction & plots every N epochs
    pred_idxs=(0, 1, 2),       # which targets to scatter-plot
    y_mean=None,               # optional de-standardization mean
    y_std=None                 # optional de-standardization std
):
    """
    Train model with live plotting + PNG saves + periodic evaluation.
    Batches can be:
      (xb, yb)  OR  (xb, pb, yb)  OR  {"x":..., "y":..., "phase":...}
    """
    device = device or next(model.parameters()).device
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)

    # Default paths inside save_dir
    if save_path is None:
        save_path = os.path.join(save_dir, "best_weights.pt")
    csv_path = os.path.join(save_dir, "learning_curves.csv")

    # ---- Resume curves from CSV if exists ----
    tr_hist, va_hist = [], []
    last_epoch = 0
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    ep_i = int(row[0]); tr = float(row[1]); va = float(row[2])
                    tr_hist.append(tr); va_hist.append(va)
                    last_epoch = ep_i
            print(f"🔁 Found CSV with {last_epoch} epochs; will resume from epoch {last_epoch+1}.")
        except Exception as e:
            print(f"⚠️ Failed to parse existing CSV; starting fresh. ({e})")
            tr_hist, va_hist, last_epoch = [], [], 0

    # Early exit if done
    if last_epoch >= epochs:
        print(f"✅ Already completed {last_epoch} ≥ target {epochs}. Restoring best and returning.")
        if os.path.exists(save_path):
            try:
                ckpt = torch.load(save_path, map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(save_path, map_location=device)
            model.load_state_dict(ckpt)
            model.to(device); model.eval()
        curves = {
            "train_total": tr_hist,
            "val_total": va_hist,
            "best_val": min(va_hist) if va_hist else float("inf"),
            "best_weights": save_path,
            "csv_path": csv_path,
        }
        return model, curves

    # ---- Init CSV header if missing ----
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_mse", "val_mse"])

    # ---- Load best weights if present (resume) ----
    if os.path.exists(save_path):
        try:
            ckpt = torch.load(save_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(save_path, map_location=device)
        try:
            model.load_state_dict(ckpt)
            model.to(device)
            print(f"🔁 Loaded best weights from {save_path}")
        except Exception as e:
            print(f"⚠️ Could not load best weights: {e}")

    # ---- Best val so far from history ----
    best_val = min(va_hist) if va_hist else float("inf")
    best_state = None

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max)
    lossf = nn.MSELoss(reduction="mean")

    # --- persistent figure setup for curves (TOTAL train/val MSE) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    (ltr,) = ax.plot(range(1, len(tr_hist)+1), tr_hist, label="Train MSE", color="C0")
    (lva,) = ax.plot(range(1, len(va_hist)+1), va_hist, label="Val MSE", color="C1")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
    ax.legend()
    fig.tight_layout()

    # ---- NEW: histories for per-target losses (for plotting) ----
    per_target_train_hist = []   # list of np.ndarray [D] per epoch
    per_target_val_hist   = []   # list of np.ndarray [D] per epoch

    def should_act(ep, control):
        if control is None:
            return False
        if isinstance(control, int):
            return ep % control == 0 or ep == epochs
        if isinstance(control, (list, tuple, set)):
            return ep in control
        return False

    # ---- Main loop ----
    for ep in range(last_epoch + 1, epochs + 1):
        # -------- Train --------
        model.train()
        train_loss, n_samples = 0.0, 0
        per_target_sse = None

        for batch in train_loader:
            xb, pb, yb = _unpack_batch(batch)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True) if pb is not None else None

            opt.zero_grad(set_to_none=True)
            pred = _forward_maybe_phase(model, xb, pb)
            loss = lossf(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

            bs = xb.size(0)
            n_samples += bs
            train_loss += loss.item() * bs

            sse_batch = _sse_per_target(pred, yb)
            per_target_sse = sse_batch if per_target_sse is None else (per_target_sse + sse_batch)

        train_loss /= max(n_samples, 1)
        tr_hist.append(train_loss)
        tr_vec = per_target_sse / max(n_samples, 1)   # Tensor [D]
        per_target_train_hist.append(tr_vec.detach().cpu().numpy())   # NEW

        # -------- Validation --------
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
        va_vec = per_target_sse / max(n_samples, 1)   # Tensor [D]
        per_target_val_hist.append(va_vec.detach().cpu().numpy())     # NEW

        sched.step()

        # -------- Save best --------
        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, save_path)

        # -------- Title text (w_pos / w_phase / w_snr if present) --------
        names = target_names if target_names else [f"t{i}" for i in range(len(tr_vec))]
        per_tr = "  ".join([f"{n}={tr_vec[i].item():.6f}" for i, n in enumerate(names)])
        per_va = "  ".join([f"{n}={va_vec[i].item():.6f}" for i, n in enumerate(names)])
        lr_now = opt.param_groups[0]["lr"]

        wpos = wphi = wsnr = None

        # --- Positional weight ---
        if hasattr(model, "w_pos"):
            wpos = model.w_pos.item() if isinstance(model.w_pos, torch.Tensor) else model.w_pos

        # --- Phase-PE weight ---
        if hasattr(model, "w_phase"):
            wphi = model.w_phase.item() if isinstance(model.w_phase, torch.Tensor) else model.w_phase

        # --- SNR encoding weight ---
        if hasattr(model, "w_snr"):
            wsnr = model.w_snr.item() if isinstance(model.w_snr, torch.Tensor) else model.w_snr

        # Build the weight-display string dynamically
        wp_list = []
        if wpos is not None:
            wp_list.append(f"w_pos={wpos:.4f}")
        if wphi is not None:
            wp_list.append(f"w_phase={wphi:.4f}")
        if wsnr is not None:
            wp_list.append(f"w_snr={wsnr:.4f}")

        wp_txt = (" | " + ", ".join(wp_list)) if wp_list else ""

        title_text = (f"Epoch {ep:03d} | Train={train_loss:.6f} | Val={val_loss:.6f} | LR={lr_now:.2e}{wp_txt}\n"
                      f"Train per-target: {per_tr}\n"
                      f"Val   per-target: {per_va}")

        # -------- Update curve plot (TOTAL) --------
        if should_act(ep, display_epochs) or should_act(ep, save_epochs):
            xs = range(1, len(tr_hist) + 1)
            ltr.set_data(xs, tr_hist)
            lva.set_data(xs, va_hist)
            ax.relim(); ax.autoscale_view()
            ax.set_title(title_text, fontsize=8)
            fig.tight_layout()

            if should_act(ep, display_epochs):
                clear_output(wait=True)
                display(fig)
                fig.canvas.draw(); fig.canvas.flush_events()

            if should_act(ep, save_epochs):
                # ======== NEW: overwrite ONE PNG for total loss ========
                out_total = os.path.join(save_dir, "curve_total_latest.png")
                fig.savefig(out_total, dpi=160)
                print(f"💾 Saved TOTAL loss curve -> {out_total}")

                # ======== NEW: per-target loss curves PNG (overwrite) ========
                try:
                    if len(per_target_train_hist) > 0:
                        arr_tr = np.vstack(per_target_train_hist)   # [E, D]
                        arr_va = np.vstack(per_target_val_hist)     # [E, D]
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
                        print(f"💾 Saved PER-TARGET loss curves -> {out_per}")
                except Exception as e:
                    print(f"⚠️ Failed to save per-target loss curves at epoch {ep}: {e}")

        # -------- Append curve row to CSV --------
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ep, train_loss, val_loss])

        if ep % print_every == 0:
            print(title_text)

        # -------- Periodic full prediction on val + scatter/metrics --------
        if eval_every and (ep % eval_every == 0 or ep == epochs):
            try:
                y_true, y_pred = _predict_on_val(model, val_loader, device)
                y_true_phys = _destandardize(y_true, y_mean, y_std)
                y_pred_phys = _destandardize(y_pred, y_mean, y_std)

                # metrics per target
                mse = ((y_true_phys - y_pred_phys) ** 2).mean(axis=0)
                mae = np.abs(y_true_phys - y_pred_phys).mean(axis=0)
                r2  = np.array([r2_score(y_true_phys[:, i], y_pred_phys[:, i])
                                for i in range(y_true_phys.shape[1])])

                # print brief metrics to console
                print("\n--- Periodic Evaluation @ epoch", ep, "---")
                for i in range(len(mse)):
                    nm = names[i] if i < len(names) else f"param_{i}"
                    print(f"  {nm:>12s} | MSE={mse[i]:.6e} | MAE={mae[i]:.6e} | R2={r2[i]:.4f}")

                # save metrics CSV
                metrics_csv = os.path.join(save_dir, f"metrics_epoch_{ep:04d}.csv")
                _write_metrics_csv(metrics_csv, ep, names, mse, mae, r2)

                # save scatter plot
                scatter_png = os.path.join(save_dir, f"pred_scatter_epoch_{ep:04d}.png")
                idxs = tuple([i for i in pred_idxs if i < y_true_phys.shape[1]])
                if len(idxs) == 0:
                    idxs = (0,)
                _plot_true_vs_pred_multi(y_true_phys, y_pred_phys, idxs=idxs,
                                         target_names=target_names, out_png=scatter_png)
                print(f"💾 Saved scatter plot -> {scatter_png}\n")
            except Exception as e:
                print(f"⚠️ Periodic evaluation failed at epoch {ep}: {e}")

    plt.close(fig)

    # -------- Restore best model at end --------
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device); model.eval()
        print(f"\n✅ Restored best model | best val MSE={best_val:.6f}")
        print(f"Saved best weights to: {save_path}")
    else:
        if os.path.exists(save_path):
            try:
                ckpt = torch.load(save_path, map_location=device, weights_only=True)
            except TypeError:
                ckpt = torch.load(save_path, map_location=device)
            try:
                model.load_state_dict(ckpt)
                model.to(device); model.eval()
                print(f"\nℹ️ Restored (existing) best weights from {save_path} | best val so far={best_val:.6f}")
            except Exception as e:
                print(f"⚠️ Could not restore existing best weights: {e}")

    curves = {
        "train_total": tr_hist,
        "val_total": va_hist,
        "best_val": best_val,
        "best_weights": save_path,
        "csv_path": csv_path,
        "last_epoch": len(tr_hist),
    }
    return model, curves
