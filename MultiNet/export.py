"""Xuất kết quả cuối cùng của một phương pháp ra định dạng chung để vẽ UMAP.

Contract cho MỌI phương pháp trong bảng so sánh (scMCC và 7 baseline):
một file .npz với đúng các khoá dưới đây. Baseline viết bằng R thì xuất CSV
rồi dùng `npz_from_csv` để chuyển — miễn là cuối cùng có cùng bộ khoá.

    emb       (N, d)  float32  — ma trận latent, d tuỳ phương pháp
    cell_ids  (N,)    str      — barcode tế bào, dùng để căn hàng giữa các file
    y_true    (N,)    str      — cell type ground truth
    y_pred    (N,)    int      — nhãn cụm do phương pháp dự đoán
    method    str              — tên hiển thị trên panel
    ari, nmi  float            — chỉ số của chính kết quả cuối cùng này
"""

import numpy as np


def save_embedding(out_path, emb, cell_ids, y_true, y_pred,
                   method, ari=None, nmi=None, **extra):
    """Ghi file .npz theo contract ở docstring module."""
    emb = np.asarray(emb, dtype=np.float32)
    cell_ids = np.asarray(cell_ids, dtype=object)
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred)

    n = emb.shape[0]
    for name, arr in (('cell_ids', cell_ids), ('y_true', y_true), ('y_pred', y_pred)):
        if len(arr) != n:
            raise ValueError(f"{name} có {len(arr)} phần tử nhưng emb có {n} hàng "
                             f"— thứ tự tế bào đã lệch, kiểm tra lại loader.")
    if len(np.unique(cell_ids)) != n:
        raise ValueError("cell_ids bị trùng — không căn hàng được giữa các phương pháp.")

    np.savez_compressed(
        out_path,
        emb=emb,
        cell_ids=cell_ids,
        y_true=y_true,
        y_pred=y_pred,
        method=str(method),
        ari=np.float64(ari if ari is not None else np.nan),
        nmi=np.float64(nmi if nmi is not None else np.nan),
        **extra,
    )
    print(f"[export] {out_path}  emb={emb.shape}  "
          f"n_types={len(np.unique(y_true))}  n_clusters={len(np.unique(y_pred))}  "
          f"ARI={ari}  NMI={nmi}")


def save_cluster_csv(out_path, cell_ids, y_pred, label_col='predicted_cell_type'):
    """Ghi CSV 2 cột: barcode | predicted_cell_type (= chỉ số cụm 0..K-1).

    `cell_ids` và `y_pred` phải cùng thứ tự — tức thứ tự của loader.dataset
    (encodeBatch duyệt shuffle=False), chính là thứ tự `rna.obs_names` trong main.
    """
    import pandas as pd

    cell_ids = np.asarray(cell_ids).astype(str)
    y_pred = np.asarray(y_pred).astype(int)
    if len(cell_ids) != len(y_pred):
        raise ValueError(f"cell_ids ({len(cell_ids)}) và y_pred ({len(y_pred)}) "
                         f"lệch độ dài — thứ tự tế bào không khớp.")
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("Barcode bị trùng — không định danh được từng tế bào.")

    df = pd.DataFrame({'barcode': cell_ids, label_col: y_pred})
    df.to_csv(out_path, index=False)
    print(f"[export] {out_path}  cells={len(df)}  n_clusters={df[label_col].nunique()}")
    return df


def csv_from_npz(npz_path, out_path):
    """Xuất CSV từ file .npz đã có (không cần train lại)."""
    d = load_embedding(npz_path)
    return save_cluster_csv(out_path, d['cell_ids'], d['y_pred'])


def load_embedding(path):
    """Đọc ngược file .npz -> dict, ép kiểu về str/float cho tiện dùng."""
    d = np.load(path, allow_pickle=True)
    return {
        'emb': d['emb'],
        'cell_ids': d['cell_ids'].astype(str),
        'y_true': d['y_true'].astype(str),
        'y_pred': d['y_pred'],
        'method': str(d['method']),
        'ari': float(d['ari']),
        'nmi': float(d['nmi']),
    }


def npz_from_csv(csv_path, out_path, method, y_true=None, cell_id_col=None,
                 pred_col='cluster', ari=None, nmi=None):
    """Chuyển output của baseline (CSV: index = barcode, các cột = chiều latent)
    sang .npz cùng contract.

    :param y_true: Series/dict ánh xạ barcode → cell type. Bỏ trống nếu CSV đã
        có cột 'cell_type'.
    :param pred_col: tên cột chứa nhãn cụm trong CSV; nếu không có thì y_pred
        được gán -1 (panel dự đoán sẽ trống, chỉ vẽ được ground truth).
    """
    import pandas as pd

    df = pd.read_csv(csv_path, index_col=0 if cell_id_col is None else cell_id_col)
    y_pred = df.pop(pred_col).to_numpy() if pred_col in df.columns else np.full(len(df), -1)
    if 'cell_type' in df.columns:
        labels = df.pop('cell_type').astype(str).to_numpy()
    else:
        if y_true is None:
            raise ValueError("CSV không có cột 'cell_type' và cũng không truyền y_true.")
        labels = pd.Series(y_true).reindex(df.index).astype(str).to_numpy()

    save_embedding(out_path, df.to_numpy(), df.index.to_numpy().astype(str),
                   labels, y_pred, method=method, ari=ari, nmi=nmi)

if __name__ == "__main__":
    # python -m scMCC.export output/scMCC_PBMC.npz output/scMCC_PBMC_clusters.csv
    import sys
    csv_from_npz(sys.argv[1], sys.argv[2])
