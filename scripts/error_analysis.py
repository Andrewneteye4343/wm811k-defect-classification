"""M7 錯誤案例分析：找出 test 集最常混淆的類別對，輸出誤判例並排圖。

流程：
  1. 載最終模型 checkpoint → 全部 test 預測
  2. 混淆矩陣找 top-2 非對角混淆對（數量最多的誤判流向）
  3. 對每對挑 3 個誤判樣本 → 並排圖（原 wafer + 標題 true→pred + 信心）

用法：
  python scripts/error_analysis.py --ckpt artifacts/runs/m6a_resnet/best_model.pt

產出：artifacts/m7_errors/confused_<true>_vs_<pred>.png（每張 3 例並排）
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wm811k import config  # noqa: E402
from wm811k.data import WM811KDataset, build_splits, labeled_frame  # noqa: E402
from wm811k.loader import load_lswmd  # noqa: E402
from wm811k.model import ResNetWafer, WaferCNN  # noqa: E402
from wm811k.metrics import confusion_matrix  # noqa: E402


def plot_error_cases(ds, indices, true_label, pred_label, confs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, i in zip(axes, indices):
        x, _ = ds[int(i)]
        ax.imshow(x[0].numpy(), cmap="gray", vmin=0, vmax=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"true={true_label} conf={confs[i]:.2f}", fontsize=9)
    fig.suptitle(f"Misclassified: true {true_label} -> pred {pred_label}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="M7 error analysis")
    ap.add_argument("--ckpt",
                    default="artifacts/runs/m6a_resnet/best_model.pt")
    ap.add_argument("--model", choices=["wafercnn", "resnet"], default=None)
    ap.add_argument("--n-pairs", type=int, default=2)
    ap.add_argument("--examples-per-pair", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="artifacts/m7_errors")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model_name = args.model or ckpt.get("config", {}).get(
        "model", "WaferCNN")
    model_cls = ResNetWafer if model_name == "ResNetWafer" else WaferCNN
    model = model_cls(num_classes=config.NUM_CLASSES)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    df = load_lswmd()
    lf = labeled_frame(df)
    tr, va, te = build_splits(lf, seed=args.seed)
    ds = WM811KDataset(lf, te)
    loader = torch.utils.data.DataLoader(ds, batch_size=512)

    print("Predicting test set ...")
    preds, targets, confs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs = torch.softmax(model(xb.to(device)), dim=1)
            preds.append(probs.argmax(dim=1).cpu())
            confs.append(probs.max(dim=1).values.cpu())
            targets.append(yb)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    confs = torch.cat(confs)

    C = confusion_matrix(targets.numpy(), preds.numpy(),
                         num_classes=config.NUM_CLASSES)
    # top 非對角混淆（排除對角）
    off = C.copy()
    np.fill_diagonal(off, 0)
    flat = [(off[r, c], r, c) for r in range(config.NUM_CLASSES)
            for c in range(config.NUM_CLASSES)]
    flat.sort(reverse=True)

    print("=== Top confusion pairs (true -> pred) ===")
    for cnt, r, c in flat[:8]:
        if cnt == 0:
            break
        print(f"  {config.FAILURE_TYPES[r]:>10} -> "
              f"{config.FAILURE_TYPES[c]:<10} {cnt:>5}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    saved = 0
    for cnt, r, c in flat[:args.n_pairs * 2]:
        if r == c or cnt < 5:
            continue
        if saved >= args.n_pairs:
            break
        true_l, pred_l = config.FAILURE_TYPES[r], config.FAILURE_TYPES[c]
        wrong = np.where((targets.numpy() == r) & (preds.numpy() == c))[0]
        picks = rng.choice(wrong, size=min(args.examples_per_pair, len(wrong)),
                           replace=False)
        fname = out_dir / f"confused_{true_l}_to_{pred_l}.png"
        plot_error_cases(ds, picks, true_l, pred_l, confs.numpy(), fname)
        print(f"  saved {fname} ({len(picks)} examples)")
        saved += 1
    print(f"Done. {saved} figures in {out_dir}/")


if __name__ == "__main__":
    main()
