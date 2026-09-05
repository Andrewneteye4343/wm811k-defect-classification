"""M7 可解釋性輸出：Grad-CAM 代表樣本熱圖（Colab/Docker 皆可跑）。

載入最終模型 checkpoint（預設 m6a_resnet）→ 對 test 集每類挑「預測正確」
的代表樣本 → 產生「原 wafer 圖 + Grad-CAM 熱圖疊加」並排 PNG。

用法：
  python scripts/gradcam_samples.py --ckpt artifacts/runs/m6a_resnet/best_model.pt
  # 選項：--model wafercnn|resnet（預設依 checkpoint 的 config 自動）、
  #       --samples-per-class 2、--classes scratch random edge-loc loc donut
  #       （預設 5 個 minority 代表類，可傳 all 做 9 類）

產出：artifacts/m7_gradcam/<class>_<idx>.png（每張 = 原圖 | 熱圖疊加並排）
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm811k import config  # noqa: E402
from wm811k.data import WM811KDataset, build_splits, labeled_frame  # noqa: E402
from wm811k.interpret import GradCAM, default_target_layer  # noqa: E402
from wm811k.loader import load_lswmd  # noqa: E402
from wm811k.model import ResNetWafer, WaferCNN  # noqa: E402


def plot_sample(x, heat, title, path):
    """並排：原 wafer 圖（0 白/1 淺灰/2 黑）| Grad-CAM 熱圖疊加。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    img = x[0].numpy()  # (32,32) 值 {0,1,2}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    # 左：wafer 原圖（灰階：0 白、1 淺灰、2 黑）
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=2)
    axes[0].set_title("wafer map (0 bg / 1 pass / 2 fail)", fontsize=9)
    # 右：熱圖疊加（jet 半透明）
    heat_cmap = LinearSegmentedColormap.from_list(
        "cam", [(0, 0, 0, 0), (0, 0, 0.6, 0.4), (1, 0, 0, 0.85)])
    axes[1].imshow(img, cmap="gray", vmin=0, vmax=2)
    axes[1].imshow(heat.numpy(), cmap=heat_cmap, vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM overlay", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M7 Grad-CAM samples")
    ap.add_argument("--ckpt", default="artifacts/runs/m6a_resnet/best_model.pt")
    ap.add_argument("--model", choices=["wafercnn", "resnet"], default=None,
                    help="預設自動從 checkpoint 的 config 判斷")
    ap.add_argument("--classes", nargs="+", default=None,
                    help="要出圖的類別（小寫）；預設 5 個 minority 代表")
    ap.add_argument("--samples-per-class", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="artifacts/m7_gradcam")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = args.model or ckpt.get("config", {}).get(
        "model", "WaferCNN")
    model_cls = ResNetWafer if model_name == "ResNetWafer" else WaferCNN
    model = model_cls(num_classes=config.NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"Loaded {model_cls.__name__} from {args.ckpt} (device={device})")

    classes = args.classes or ["scratch", "random", "edge-loc", "loc", "donut"]
    if classes == ["all"]:
        classes = config.FAILURE_TYPES

    # ── test 集（隨機切分 — 與訓練一致）──
    df = load_lswmd()
    lf = labeled_frame(df)
    tr, va, te = build_splits(lf, seed=args.seed)
    ds = WM811KDataset(lf, te)
    loader = torch.utils.data.DataLoader(ds, batch_size=512)

    # 全 test 預測（找每類「正確」樣本）
    print("Predicting test set ...")
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu())
            targets.append(yb)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    correct_idx = {c: [] for c in range(config.NUM_CLASSES)}
    for i in range(len(targets)):
        if preds[i] == targets[i]:
            correct_idx[int(targets[i])].append(i)
    print(f"correct per class: "
          f"{[len(correct_idx[c]) for c in range(config.NUM_CLASSES)]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model, default_target_layer(model))
    rng = np.random.default_rng(args.seed)
    saved = 0
    for cname in classes:
        cidx = config.LABEL_TO_IDX[cname]
        pool = correct_idx[cidx]
        if len(pool) == 0:
            print(f"  {cname}: 無正確樣本，跳過")
            continue
        picks = rng.choice(pool, size=min(args.samples_per_class, len(pool)),
                           replace=False)
        for n, i in enumerate(picks):
            x, y = ds[int(i)]
            xb = x.unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(xb)[0]
            prob = torch.softmax(logit, dim=0)
            conf = float(prob[cidx])
            # 注意：dataset 的 x 未歸一化（{0,1,2}）— 模型訓練也吃未歸一化
            heat = cam.heatmap(xb, class_idx=cidx).cpu()
            fname = out_dir / f"{cname}_ex{n}.png"
            title = f"{cname} (true={cname}) conf={conf:.2f}"
            plot_sample(x, heat, title, fname)
            saved += 1
            print(f"  saved {fname}")
    print(f"Done. {saved} images in {out_dir}/")


if __name__ == "__main__":
    main()
