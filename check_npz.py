"""Kiểm tra một file .npz có ghép được với file .npz của scMCC hay không.

    python check_npz.py --ref scMCC_PBMC.npz --cand scMDC_PBMC.npz
"""
import argparse
import sys

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Console Windows mặc định cp1252 → print tiếng Việt sẽ lỗi.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

REQUIRED = ("emb", "cell_ids", "y_true", "y_pred", "method", "ari", "nmi")


def load(path):
    d = np.load(path, allow_pickle=True)
    missing = [k for k in REQUIRED if k not in d.files]
    assert not missing, f"{path}: thiếu khoá {missing}"
    return {k: d[k] for k in d.files}


ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="File .npz của scMCC (chuẩn đối chiếu)")
ap.add_argument("--cand", required=True, help="File .npz cần kiểm tra")
args = ap.parse_args()
ref, cand = load(args.ref), load(args.cand)

# 1. Hình dạng & kiểu dữ liệu
emb = cand["emb"]
n = emb.shape[0]
assert emb.ndim == 2 and emb.shape[1] >= 2, f"emb phải là (N, d>=2), đang là {emb.shape}"
assert np.isfinite(emb).all(), "emb chứa NaN/Inf — UMAP sẽ lỗi"
for k in ("cell_ids", "y_true", "y_pred"):
    assert len(cand[k]) == n, f"{k} có {len(cand[k])} phần tử, emb có {n} hàng"
ids = cand["cell_ids"].astype(str)
assert len(np.unique(ids)) == n, "cell_ids bị trùng"
assert np.issubdtype(cand["y_pred"].dtype, np.integer), f"y_pred phải là int, đang là {cand['y_pred'].dtype}"
assert not (cand["y_pred"] == -1).all(), "y_pred toàn -1 — chưa có nhãn cụm"

print(f"method={cand['method']}  N={n}  d={emb.shape[1]}  "
      f"K_pred={len(np.unique(cand['y_pred']))}  K_true={len(np.unique(cand['y_true']))}")

# 2. Căn hàng với scMCC theo barcode
ref_ids = ref["cell_ids"].astype(str)
common, i_ref, i_cand = np.intersect1d(ref_ids, ids, return_indices=True)
print(f"barcode chung với ref: {len(common)} "
      f"(= {len(common) / len(ref_ids):.1%} của ref, {len(common) / n:.1%} của cand)")
assert len(common) > 0, "Không có barcode chung — barcode bị đổi định dạng?"
if len(common) < 0.9 * len(ref_ids):
    print("[WARN] mất >10% tế bào so với scMCC — kiểm tra bước lọc tế bào / định dạng barcode")

# 3. Ground truth phải trùng khớp trên phần giao
bad = (ref["y_true"].astype(str)[i_ref] != cand["y_true"].astype(str)[i_cand]).sum()
assert bad == 0, f"{bad} tế bào có y_true khác scMCC — sai cột nhãn hoặc đang lưu mã số thay vì tên"

# 4. ARI/NMI lưu trong file phải khớp với chính y_pred
yt, yp = cand["y_true"].astype(str), cand["y_pred"]
ari, nmi = adjusted_rand_score(yt, yp), normalized_mutual_info_score(yt, yp)
print(f"ARI tính lại={ari:.4f} (trong file {float(cand['ari']):.4f}) | "
      f"NMI tính lại={nmi:.4f} (trong file {float(cand['nmi']):.4f})")
if not (np.isclose(ari, float(cand["ari"]), atol=1e-3) and np.isclose(nmi, float(cand["nmi"]), atol=1e-3)):
    print("[WARN] ARI/NMI trong file không khớp với y_pred — có thể đang lưu số của checkpoint khác")

print("OK")
