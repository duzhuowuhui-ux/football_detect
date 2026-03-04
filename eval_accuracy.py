"""
预测精度评估脚本
================
输入：预测掩膜 .tif  +  真值标注 .shp
输出：IoU / Precision / Recall / F1 / OA + 混淆矩阵 + 可视化图
"""

import numpy as np
import sys
from pathlib import Path

# ══════════════════════════════════════════════════
# ★ 修改这里
# ══════════════════════════════════════════════════
PRED_MASK  = r"/root/autodl-tmp/预测结果/北京/北京_mask.tif" # 预测掩膜
GT_SHP     = r"/root/autodl-tmp/shp/北京/Beijing1.shp"    # 真值 shapefile
CLASS_FIELD = 'class'    # shp 中表示类别的字段名；若无该字段则全部视为前景，改为 None
OUT_DIR    = r'/root/autodl-tmp/精度评定结果/北京'        # 结果保存目录
# ══════════════════════════════════════════════════


def check_deps():
    missing = []
    for pkg in ['rasterio', 'geopandas', 'shapely']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[错误] 缺少依赖，请运行：pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import rasterio
import geopandas as gpd
from rasterio.features import rasterize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json


# ════════════════════════════════════════════════
# 1. 读取预测掩膜
# ════════════════════════════════════════════════
def load_pred(path: str) -> tuple:
    with rasterio.open(path) as src:
        pred      = src.read(1).astype(np.uint8)   # (H, W)  值 0/1
        profile   = src.profile
        transform = src.transform
        crs       = src.crs
        h, w      = src.height, src.width
    print(f"[预测掩膜]  尺寸={h}×{w}  CRS={crs}")
    print(f"           前景像素={pred.sum():,}  占比={pred.sum()/(h*w)*100:.3f}%")
    return pred, profile, transform, crs, (h, w)


# ════════════════════════════════════════════════
# 2. 将 shapefile 光栅化为真值掩膜
# ════════════════════════════════════════════════
def rasterize_gt(shp_path: str, transform, crs, shape: tuple,
                 class_field: str) -> np.ndarray:
    gdf = gpd.read_file(shp_path)
    print(f"\n[真值标注]  多边形数={len(gdf)}  CRS={gdf.crs}")
    print(f"           字段列表={list(gdf.columns)}")

    # 坐标系对齐
    if gdf.crs != crs:
        print(f"           ⚠ CRS 不一致，自动转换 {gdf.crs} → {crs}")
        gdf = gdf.to_crs(crs)

    # 构建 (geometry, value) 对
    if class_field and class_field in gdf.columns:
        shapes = [(geom, int(val))
                  for geom, val in zip(gdf.geometry, gdf[class_field])
                  if geom is not None]
        print(f"           使用字段 '{class_field}' 作为类别值")
    else:
        # 无类别字段，全部前景值=1
        shapes = [(geom, 1)
                  for geom in gdf.geometry if geom is not None]
        print(f"           未找到字段 '{class_field}'，全部标记为前景(1)")

    gt = rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    print(f"           光栅化完成  前景像素={gt.sum():,}  占比={gt.sum()/(shape[0]*shape[1])*100:.3f}%")
    return gt


# ════════════════════════════════════════════════
# 3. 计算指标
# ════════════════════════════════════════════════
def calc_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred_b = (pred > 0).astype(np.uint8)
    gt_b   = (gt   > 0).astype(np.uint8)

    TP = int(((pred_b == 1) & (gt_b == 1)).sum())
    FP = int(((pred_b == 1) & (gt_b == 0)).sum())
    FN = int(((pred_b == 0) & (gt_b == 1)).sum())
    TN = int(((pred_b == 0) & (gt_b == 0)).sum())

    total = TP + FP + FN + TN

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    iou       = TP / (TP + FP + FN + 1e-9)
    oa        = (TP + TN) / total
    miou      = (iou + TN / (TN + FP + FN + 1e-9)) / 2   # 含背景 mIoU

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'Precision': precision,
        'Recall':    recall,
        'F1':        f1,
        'IoU':       iou,
        'OA':        oa,
        'mIoU':      miou,
        'total_px':  total,
    }


def print_metrics(m: dict):
    print("\n" + "="*50)
    print("  精度评估结果")
    print("="*50)
    print(f"  TP（正确检测）: {m['TP']:>12,} 像素")
    print(f"  FP（误检）    : {m['FP']:>12,} 像素")
    print(f"  FN（漏检）    : {m['FN']:>12,} 像素")
    print(f"  TN（正确背景）: {m['TN']:>12,} 像素")
    print("-"*50)
    print(f"  IoU           : {m['IoU']:.4f}")
    print(f"  Precision     : {m['Precision']:.4f}")
    print(f"  Recall        : {m['Recall']:.4f}")
    print(f"  F1            : {m['F1']:.4f}")
    print(f"  OA            : {m['OA']:.4f}")
    print(f"  mIoU          : {m['mIoU']:.4f}")
    print("="*50)

    # 简单分析
    print("\n  结果解读：")
    if m['IoU'] >= 0.7:
        print(f"  ✓ IoU={m['IoU']:.3f} 检测效果良好")
    elif m['IoU'] >= 0.4:
        print(f"  △ IoU={m['IoU']:.3f} 有一定效果但较弱，建议微调模型")
    else:
        print(f"  ✗ IoU={m['IoU']:.3f} 效果较差，建议标注数据后微调模型")

    if m['Recall'] < 0.5:
        fp_pct = m['FP'] / (m['TP'] + m['FP'] + 1e-9) * 100
        print(f"  ✗ 漏检严重（Recall={m['Recall']:.3f}），模型对该城市保守，建议降低阈值或微调")
    if m['Precision'] < 0.5:
        print(f"  ✗ 误检较多（Precision={m['Precision']:.3f}），建议提高阈值或微调模型")


# ════════════════════════════════════════════════
# 4. 可视化
# ════════════════════════════════════════════════
def visualize(pred: np.ndarray, gt: np.ndarray, m: dict, out_dir: Path):
    # 错误图：0=正确背景  1=TP  2=FP(误检)  3=FN(漏检)
    err = np.zeros_like(pred, dtype=np.uint8)
    err[(pred==1) & (gt==1)] = 1   # TP  绿
    err[(pred==1) & (gt==0)] = 2   # FP  红
    err[(pred==0) & (gt==1)] = 3   # FN  黄

    # 降采样（大图直接显示太慢）
    step = max(1, max(pred.shape) // 2000)
    pred_s = pred[::step, ::step]
    gt_s   = gt[::step, ::step]
    err_s  = err[::step, ::step]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"精度评估  IoU={m['IoU']:.4f}  P={m['Precision']:.4f}  R={m['Recall']:.4f}  F1={m['F1']:.4f}",
        fontsize=13, fontweight='bold'
    )

    # 预测conda --version
    axes[0].imshow(pred_s, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title('预测掩膜'); axes[0].axis('off')

    # 真值
    axes[1].imshow(gt_s, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title('真值标注'); axes[1].axis('off')

    # 错误分析
    cmap_err = plt.cm.colors.ListedColormap(['#1a1a1a', '#2ecc71', '#e74c3c', '#f39c12'])
    axes[2].imshow(err_s, cmap=cmap_err, vmin=0, vmax=3, interpolation='nearest')
    axes[2].set_title('误差分析')
    axes[2].axis('off')
    patches = [
        mpatches.Patch(color='#2ecc71', label=f'TP 正确检测 {m["TP"]:,}'),
        mpatches.Patch(color='#e74c3c', label=f'FP 误检     {m["FP"]:,}'),
        mpatches.Patch(color='#f39c12', label=f'FN 漏检     {m["FN"]:,}'),
    ]
    axes[2].legend(handles=patches, loc='lower right', fontsize=8)

    plt.tight_layout()
    out_path = out_dir / 'eval_visualization.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  可视化图: {out_path}")


# ════════════════════════════════════════════════
# 主流程
# ════════════════════════════════════════════════
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("="*50)
    print("  足球场检测精度评估")
    print("="*50)

    # 1. 加载预测掩膜
    pred, profile, transform, crs, shape = load_pred(PRED_MASK)

    # 2. 光栅化真值
    gt = rasterize_gt(GT_SHP, transform, crs, shape, CLASS_FIELD)

    # 3. 计算指标
    m = calc_metrics(pred, gt)
    print_metrics(m)

    # 4. 保存 JSON
    json_path = out_dir / 'eval_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"  指标 JSON : {json_path}")

    # 5. 可视化
    visualize(pred, gt, m, out_dir)

    print("\n完成！")


if __name__ == '__main__':
    main()
