"""
足球场遥感检测系统 - 跨城市预测脚本
=====================================
用已训练的武汉模型对任意城市的遥感影像进行足球场检测。

直接运行（无需任何命令行参数）：
    python predict_city.py
"""

import sys
import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 依赖检查 ─────────────────────────────────────────────────────
def _check_deps():
    missing = []
    try:
        import rasterio
    except ImportError:
        missing.append("rasterio")
    if missing:
        print(f"[错误] 缺少依赖，请运行：pip install {' '.join(missing)}")
        sys.exit(1)

_check_deps()
import rasterio
from rasterio.transform import from_bounds


# ════════════════════════════════════════════════════════════════
# 模型定义（与 model.py 完全一致，单文件独立运行）
# ════════════════════════════════════════════════════════════════

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=True),
            nn.InstanceNorm2d(mid_ch, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, dilations=None):
        super().__init__()
        dilations = dilations or [1, 6, 12, 18]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True), nn.ReLU(inplace=True))
        self.atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=True),
                nn.InstanceNorm2d(out_ch, affine=True), nn.ReLU(inplace=True))
            for d in dilations])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.ReLU(inplace=True))          # ← GAP后不加Norm
        n = 1 + len(dilations) + 1
        self.proj = nn.Sequential(
            nn.Conv2d(n * out_ch, out_ch, 1, bias=True),
            nn.InstanceNorm2d(out_ch, affine=True), nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        gap  = F.interpolate(self.gap(x), (h, w), mode='bilinear', align_corners=False)
        parts = [self.conv1(x)] + [b(x) for b in self.atrous] + [gap]
        return self.proj(torch.cat(parts, dim=1))


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g,   F_int, 1, bias=True), nn.InstanceNorm2d(F_int, affine=True))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l,   F_int, 1, bias=True), nn.InstanceNorm2d(F_int, affine=True))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1,     1, bias=True), nn.InstanceNorm2d(1,     affine=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, x1.shape[-2:], mode='bilinear', align_corners=False)
        return x * self.psi(self.relu(g1 + x1))


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, use_attention=True):
        super().__init__()
        self.enc1  = DoubleConv(in_channels, 64);  self.pool1 = nn.MaxPool2d(2)
        self.enc2  = DoubleConv(64, 128);           self.pool2 = nn.MaxPool2d(2)
        self.enc3  = DoubleConv(128, 256);          self.pool3 = nn.MaxPool2d(2)
        self.enc4  = DoubleConv(256, 512);          self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = ASPP(512, 256)
        self.up4   = nn.ConvTranspose2d(256, 256, 2, 2)
        self.attn4 = AttentionGate(256, 512, 256) if use_attention else None
        self.dec4  = DoubleConv(256+512, 256)
        self.up3   = nn.ConvTranspose2d(256, 128, 2, 2)
        self.attn3 = AttentionGate(128, 256, 128) if use_attention else None
        self.dec3  = DoubleConv(128+256, 128)
        self.up2   = nn.ConvTranspose2d(128, 64, 2, 2)
        self.attn2 = AttentionGate(64, 128, 64) if use_attention else None
        self.dec2  = DoubleConv(64+128, 64)
        self.up1   = nn.ConvTranspose2d(64, 32, 2, 2)
        self.attn1 = AttentionGate(32, 64, 32) if use_attention else None
        self.dec1  = DoubleConv(32+64, 32)
        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def _decode(self, up, attn, dec, x, skip):
        x = up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=False)
        if attn is not None:
            skip = attn(g=x, x=skip)
        return dec(torch.cat([x, skip], dim=1))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self._decode(self.up4, self.attn4, self.dec4, b,  e4)
        d3 = self._decode(self.up3, self.attn3, self.dec3, d4, e3)
        d2 = self._decode(self.up2, self.attn2, self.dec2, d3, e2)
        d1 = self._decode(self.up1, self.attn1, self.dec1, d2, e1)
        return self.out_conv(d1)


# ════════════════════════════════════════════════════════════════
# 核心预测器
# ════════════════════════════════════════════════════════════════

class CityPredictor:
    """
    对任意城市四波段遥感影像进行足球场检测。

    参数
    ----
    model_path   : 训练好的模型文件路径（best_model.pth）
    tile_size    : 滑动窗口切片大小，与训练时一致（默认 256）
    overlap      : 切片重叠像素数（默认 64），减少拼接边界伪影
    threshold    : 二值化阈值，0.5 较保守；调低（如 0.3）可检出更多但误检增加
    max_value    : 影像归一化分母，与训练时一致（默认 2047，适合 11bit 影像）
    band_order   : 读取的波段索引列表，默认前 4 个波段 [1,2,3,4]
    device       : 'cuda' / 'cpu' / 'auto'（自动检测）
    """

    def __init__(
        self,
        model_path: str,
        tile_size:  int   = 256,
        overlap:    int   = 64,
        threshold:  float = 0.5,
        max_value:  float = 2047.0,
        band_order: list  = None,
        device:     str   = 'auto',
    ):
        self.tile_size  = tile_size
        self.overlap    = overlap
        self.threshold  = threshold
        self.max_value  = max_value
        self.band_order = band_order or [1, 2, 3, 4]

        # 设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"[设备] 使用 {self.device}")

        # 加载模型
        self._load_model(model_path)

    # ── 模型加载 ──────────────────────────────────────────────────
    def _load_model(self, path: str):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在：{path}")

        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt.get('config', {})

        in_ch       = cfg.get('input_channels', 4)
        n_cls       = cfg.get('num_classes',    2)
        use_attn    = cfg.get('use_attention',  True)

        self.model = ImprovedUNet(in_ch, n_cls, use_attention=use_attn).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

        self.num_classes = n_cls
        train_iou = ckpt.get('iou', 0)
        train_ep  = ckpt.get('epoch', '?')
        print(f"[模型] 加载成功  训练 IoU={train_iou:.4f}  Epoch={train_ep}")
        print(f"[模型] 类别数={n_cls}  输入通道={in_ch}  注意力门控={use_attn}")

    def _read_image(self, image_path: str):
        # ── 武汉训练集的每个波段统计（从训练数据计算得到）──────────
        # 格式：[B1均值, B2均值, B3均值, B4均值]
        # 如果不知道可以先运行下面的 calc_wuhan_stats.py 计算
        WUHAN_MEAN = [0.007, 0.009, 0.007, 0.014]   # 归一化后的武汉均值
        WUHAN_STD  =  [0.006, 0.007, 0.006, 0.009]   # 归一化后的武汉标准差

        image_path = Path(image_path)
        print(f"\n[读取] {image_path.name}")
        with rasterio.open(image_path) as src:
            n_bands = src.count
            print(f"  影像尺寸: {src.height} × {src.width}")
            print(f"  波段数量: {n_bands}")

            bands_to_read = []
            for b in self.band_order:
                if b <= n_bands:
                    bands_to_read.append(src.read(b).astype(np.float32))
                else:
                    print(f"  ⚠ 波段 {b} 不存在，用零填充")
                    bands_to_read.append(np.zeros((src.height, src.width), dtype=np.float32))

            self.profile   = src.profile.copy()
            self.transform = src.transform
            self.crs       = src.crs

        image = np.stack(bands_to_read, axis=-1)  # (H, W, C)

        # ── 第1步：百分位归一化到 [0,1] ────────────────────────────
        for c in range(image.shape[2]):
            lo = np.percentile(image[:, :, c], 1)
            hi = np.percentile(image[:, :, c], 99)
            image[:, :, c] = np.clip((image[:, :, c] - lo) / (hi - lo + 1e-8), 0, 1)

        # ── 第2步：均值方差对齐到武汉分布 ──────────────────────────
        for c in range(min(image.shape[2], len(WUHAN_MEAN))):
            city_mean = image[:, :, c].mean()
            city_std  = image[:, :, c].std() + 1e-8
            # 先标准化，再映射到武汉分布
            image[:, :, c] = (image[:, :, c] - city_mean) / city_std * WUHAN_STD[c] + WUHAN_MEAN[c]
        image = np.clip(image, 0, 1)

        print(f"  直方图匹配完成（对齐到武汉分布）")
        return image
    # ── 新增：TTA 方法（加在 _read_image 后面）──────────────────────
    def _predict_tile_tta(self, tile: np.ndarray) -> np.ndarray:
        """
        D4 对称群 8 方向预测取均值（Moshkov et al., Scientific Reports 2020）
        tile: (C, H, W)，返回 (H, W) 前景概率
        """
        augments = [(0,False),(1,False),(2,False),(3,False),
                    (0,True), (1,True), (2,True), (3,True)]

        def aug(t, k, flip):
            t = np.rot90(t, k=k, axes=(1, 2))
            if flip: t = t[:, :, ::-1]
            return t.copy()

        def deaug(p, k, flip):
            if flip: p = p[:, ::-1]
            return np.rot90(p, k=-k, axes=(0, 1)).copy()

        batch = np.stack([aug(tile, k, f) for k, f in augments])
        t = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type,
                                    enabled=(self.device.type=='cuda')):
                logits = self.model(t)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().float().numpy()
        deauged = np.stack([deaug(probs[i], k, f) for i,(k,f) in enumerate(augments)])
        return deauged.mean(axis=0)
    # ── 滑动窗口推理 ──────────────────────────────────────────────
    def _sliding_window_predict(self, image: np.ndarray) -> tuple:
        """
        滑动窗口推理，重叠区域取概率均值。
        返回：(预测类别图 uint8, 前景概率图 float32)
        """
        H, W = image.shape[:2]
        T    = self.tile_size
        step = T - self.overlap
        n    = self.num_classes

        # 边缘填充，使影像可被完整覆盖
        pad_h = (T - H % T) % T if H % step != 0 else 0
        pad_w = (T - W % step) % step if W % step != 0 else 0
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        PH, PW = image.shape[:2]
        prob_sum  = np.zeros((PH, PW, n), dtype=np.float32)
        count_map = np.zeros((PH, PW),    dtype=np.float32)

        # 生成切片坐标列表
        ys = list(range(0, PH - T + 1, step))
        xs = list(range(0, PW - T + 1, step))
        positions = [(y, x) for y in ys for x in xs]
        total = len(positions)

        print(f"\n[推理] 切片大小={T}  步长={step}  共 {total} 个切片")

        # 批量推理（减少 GPU 数据传输次数）
        BATCH = 8  # 每次推理的切片数，可按显存调整
        tiles_buf, pos_buf = [], []

        def _flush(tiles_buf, pos_buf):
            if not tiles_buf:
                return
            batch_t = torch.from_numpy(
                np.stack(tiles_buf)        # (B, C, T, T)
            ).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type=='cuda')):
                    logits = self.model(batch_t)
                probs = F.softmax(logits, dim=1).permute(0,2,3,1).cpu().float().numpy()

            for k, (y, x) in enumerate(pos_buf):
                prob_sum[y:y+T, x:x+T]  += probs[k]
                count_map[y:y+T, x:x+T] += 1.0

# 原版 _flush（批量推理）改为 TTA 逐 tile 推理
# 把 with tqdm... 那个循环改成：
        with tqdm(total=total, desc="  推理中", ncols=80, unit="tile") as pbar:
            for y, x in positions:
                tile = image[y:y+T, x:x+T].transpose(2, 0, 1).astype(np.float32)
                fg   = self._predict_tile_tta(tile)          # (T, T)
                prob_sum[y:y+T, x:x+T, 1] += fg
                prob_sum[y:y+T, x:x+T, 0] += 1.0 - fg
                count_map[y:y+T, x:x+T]   += 1.0
                pbar.update(1)

            _flush(tiles_buf, pos_buf)  # 处理剩余

        # 均值融合
        count_map = np.maximum(count_map, 1.0)
        prob_sum /= count_map[:, :, np.newaxis]

        # 裁回原始尺寸
        prob_sum = prob_sum[:H, :W]
        fg_prob  = prob_sum[:, :, 1]                                    # 前景概率
        pred     = (fg_prob >= self.threshold).astype(np.uint8)         # 二值掩膜

        return pred, fg_prob

    # ── 连通域分析 ────────────────────────────────────────────────
    def _count_fields(self, mask: np.ndarray, min_area_px: int = 50) -> dict:
        """统计检测到的足球场数量，过滤面积过小的噪声连通域"""
        try:
            import cv2
        except ImportError:
            print("  ⚠ 未安装 opencv，跳过连通域统计")
            return {"count": -1, "areas_px": []}

        mask_u8 = (mask * 255).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        areas = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area >= min_area_px:
                areas.append(area)

        return {
            "count":      len(areas),
            "areas_px":   sorted(areas, reverse=True),
            "total_px":   int(mask.sum()),
        }

    # ── 结果保存 ──────────────────────────────────────────────────
    def _save_results(self, pred: np.ndarray, fg_prob: np.ndarray,
                      out_dir: Path, city_name: str):
        out_dir.mkdir(parents=True, exist_ok=True)

        # 更新 profile
        profile = self.profile.copy()
        profile.update(count=1, compress='lzw')

        # 1. 二值掩膜（0/1）
        mask_path = out_dir / f"{city_name}_mask.tif"
        profile.update(dtype=rasterio.uint8)
        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(pred.astype(np.uint8), 1)
        print(f"  ✓ 二值掩膜  : {mask_path}")

        # 2. 前景概率图（0.0~1.0）
        prob_path = out_dir / f"{city_name}_probability.tif"
        profile.update(dtype=rasterio.float32)
        with rasterio.open(prob_path, 'w', **profile) as dst:
            dst.write(fg_prob.astype(np.float32), 1)
        print(f"  ✓ 概率图    : {prob_path}")

        return mask_path, prob_path

    # ── 可视化 ────────────────────────────────────────────────────
    def _visualize(self, image: np.ndarray, pred: np.ndarray,
                   fg_prob: np.ndarray, out_dir: Path, city_name: str):
        """生成 RGB 叠加图和概率热力图"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            print("  ⚠ 未安装 matplotlib，跳过可视化")
            return

        # 取 RGB 波段（前3通道）拉伸显示
        rgb = image[:, :, :3].copy()
        lo, hi = np.percentile(rgb, [2, 98])
        rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'{city_name} - 足球场检测结果', fontsize=14, fontweight='bold')

        # 左：原始影像
        axes[0].imshow(rgb)
        axes[0].set_title('原始影像 (RGB)'); axes[0].axis('off')

        # 中：检测掩膜叠加
        axes[1].imshow(rgb)
        overlay = np.zeros((*pred.shape, 4), dtype=np.float32)
        overlay[pred == 1] = [0.0, 0.8, 0.2, 0.5]   # 绿色半透明
        axes[1].imshow(overlay)
        patch = mpatches.Patch(color=(0, 0.8, 0.2, 0.8), label='足球场')
        axes[1].legend(handles=[patch], loc='lower right', fontsize=9)
        axes[1].set_title('检测结果叠加'); axes[1].axis('off')

        # 右：前景概率热力图
        cmap = LinearSegmentedColormap.from_list('prob', ['#0d0d0d', '#1a9850', '#fdae61', '#d73027'])
        im = axes[2].imshow(fg_prob, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[2], fraction=0.046, label='足球场概率')
        axes[2].set_title('前景概率热力图'); axes[2].axis('off')

        plt.tight_layout()
        vis_path = out_dir / f"{city_name}_visualization.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 可视化图  : {vis_path}")

    # ── 主入口 ────────────────────────────────────────────────────
    def predict(
        self,
        image_path:  str,
        out_dir:     str  = "./prediction_output",
        city_name:   str  = None,
        min_area_px: int  = 50,
        visualize:   bool = True,
    ) -> dict:
        """
        对单幅影像进行足球场检测。

        参数
        ----
        image_path  : 输入 GeoTIFF 路径
        out_dir     : 结果输出目录
        city_name   : 城市名（用于输出文件命名，None 时自动用文件名）
        min_area_px : 最小连通域面积（像素），过小的视为噪声
        visualize   : 是否生成可视化图像

        返回
        ----
        dict: 包含检测统计和输出文件路径
        """
        t0 = time.time()

        print("\n" + "="*60)
        print("  足球场跨城市检测")
        print("="*60)

        city_name = city_name or Path(image_path).stem
        out_dir   = Path(out_dir) / city_name

        # 1. 读取影像
        image = self._read_image(image_path)
        H, W  = image.shape[:2]

        # 2. 滑动窗口推理
        pred, fg_prob = self._sliding_window_predict(image)

        # 3. 统计
        stats = self._count_fields(pred, min_area_px)
        fg_ratio = pred.sum() / (H * W) * 100
        print(f"\n[统计] 检测到足球场  : {stats['count']} 个")
        print(f"[统计] 前景像素占比  : {fg_ratio:.3f}%")
        print(f"[统计] 前景总像素数  : {stats['total_px']:,}")
        if stats['areas_px']:
            areas = stats['areas_px']
            print(f"[统计] 最大场地面积  : {areas[0]:,} 像素")
            print(f"[统计] 最小场地面积  : {areas[-1]:,} 像素")
            print(f"[统计] 平均场地面积  : {np.mean(areas):.0f} 像素")

        # 4. 保存 GeoTIFF
        print(f"\n[保存] 结果目录: {out_dir}")
        mask_path, prob_path = self._save_results(pred, fg_prob, out_dir, city_name)

        # 5. 可视化
        if visualize:
            self._visualize(image, pred, fg_prob, out_dir, city_name)

        # 6. 保存 JSON 统计
        result = {
            "city":         city_name,
            "image":        str(Path(image_path).resolve()),
            "image_size":   [H, W],
            "threshold":    self.threshold,
            "fields_count": stats['count'],
            "total_fg_px":  stats['total_px'],
            "fg_ratio_pct": float(f"{fg_ratio:.4f}"),
            "areas_px":     stats['areas_px'][:20],   # 最多记录前20个
            "outputs": {
                "mask":          str(mask_path),
                "probability":   str(prob_path),
            },
            "elapsed_sec": round(time.time() - t0, 1),
        }
        json_path = out_dir / f"{city_name}_stats.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 统计信息  : {json_path}")

        print(f"\n[完成] 总耗时 {result['elapsed_sec']}s")
        print("="*60)
        return result


# ════════════════════════════════════════════════════════════════
# 命令行入口
# ════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="足球场跨城市遥感检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本用法（自动检测设备）
  python predict_city.py --image beijing.tif --model best_model.pth

  # 指定城市名和输出目录
  python predict_city.py --image shanghai.tif --model best_model.pth \\
      --city 上海 --out ./results

  # 调低阈值（更激进检测，提高召回率但误检增加）
  python predict_city.py --image guangzhou.tif --model best_model.pth \\
      --threshold 0.3

  # 只有 3 波段 RGB 影像（无近红外），精度会下降
  python predict_city.py --image rgb_only.tif --model best_model.pth \\
      --bands 1 2 3 0

  # 批量预测多个城市
  python predict_city.py --batch cities.txt --model best_model.pth
        """
    )
    parser.add_argument('--image',     type=str,   help='输入 GeoTIFF 影像路径')
    parser.add_argument('--model',     type=str,   required=True, help='模型文件路径（best_model.pth）')
    parser.add_argument('--out',       type=str,   default='./prediction_output', help='输出目录（默认 ./prediction_output）')
    parser.add_argument('--city',      type=str,   default=None,  help='城市名（默认用影像文件名）')
    parser.add_argument('--threshold', type=float, default=0.5,   help='二值化阈值 0~1（默认 0.5）')
    parser.add_argument('--tile',      type=int,   default=256,   help='切片大小（默认 256，需与训练一致）')
    parser.add_argument('--overlap',   type=int,   default=64,    help='切片重叠像素（默认 64）')
    parser.add_argument('--max_val',   type=float, default=2047.0,help='归一化最大值（默认 2047）')
    parser.add_argument('--bands',     type=int,   nargs='+',     default=[1,2,3,4], help='读取的波段索引（默认 1 2 3 4）')
    parser.add_argument('--device',    type=str,   default='auto',help='设备 auto/cuda/cpu（默认 auto）')
    parser.add_argument('--min_area',  type=int,   default=50,    help='最小连通域像素数（默认 50）')
    parser.add_argument('--no_vis',    action='store_true',        help='不生成可视化图像')
    parser.add_argument('--batch',     type=str,   default=None,  help='批量预测：包含影像路径的文本文件（每行一个路径）')
    return parser.parse_args()


def main():
    # ================================================================
    # ★ 在这里直接修改路径，之后直接运行脚本，无需命令行参数
    # ================================================================

    # 模型路径（必填）
    MODEL_PATH  = r"/root/autodl-tmp/output/models_binary/best_model.pth"

    # 要预测的影像路径（单张）
    IMAGE_PATH  = r"/root/autodl-tmp/image/Beijing.tif"

    # 城市名（用于输出文件命名，改成对应城市即可）
    CITY_NAME   = '北京'

    # 结果保存目录
    OUT_DIR     = r"/root/autodl-tmp/预测结果"

    # ── 高级参数（一般不需要改）────────────────────────────────────
    THRESHOLD   = 0.30      # 检测阈值：调低(0.3)召回更多，调高(0.7)更保守
    TILE_SIZE   = 256      # 切片大小，须与训练一致
    OVERLAP     = 64       # 切片重叠像素
    MAX_VALUE   = 2047.0   # 归一化分母：8bit影像改255，16bit改65535
    BANDS       = [1,2,3,4]# 读取的波段顺序，只有RGB时改为 [1,2,3,0]
    MIN_AREA    = 50       # 小于此像素数的连通域视为噪声，直接过滤
    VISUALIZE   = True     # 是否生成可视化图（False可加快速度）
    DEVICE      = 'auto'   # 'auto' 自动选GPU/CPU，或指定 'cuda' / 'cpu'

    # ── 批量预测多个城市（如不需要，保持 BATCH_LIST 为空列表即可）──
    # 填入多个 (影像路径, 城市名) 的元组，会依次处理所有城市
    BATCH_LIST  = [
        # ('/root/autodl-tmp/data/beijing.tif',   '北京'),
        # ('/root/autodl-tmp/data/shanghai.tif',  '上海'),
        # ('/root/autodl-tmp/data/guangzhou.tif', '广州'),
    ]
    # ================================================================

    # 创建预测器
    predictor = CityPredictor(
        model_path = MODEL_PATH,
        tile_size  = TILE_SIZE,
        overlap    = OVERLAP,
        threshold  = THRESHOLD,
        max_value  = MAX_VALUE,
        band_order = BANDS,
        device     = DEVICE,
    )

    # ── 批量模式（BATCH_LIST 不为空时自动进入）───────────────────
    if BATCH_LIST:
        all_results = []
        print(f"[批量] 共 {len(BATCH_LIST)} 个影像")
        for img_path, city in BATCH_LIST:
            try:
                r = predictor.predict(
                    image_path  = img_path,
                    out_dir     = OUT_DIR,
                    city_name   = city,
                    min_area_px = MIN_AREA,
                    visualize   = VISUALIZE,
                )
                all_results.append(r)
            except Exception as e:
                print(f"  ✗ {city} ({img_path}) 失败：{e}")

        summary_path = Path(OUT_DIR) / 'batch_summary.json'
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[汇总] 批量结果已保存：{summary_path}")

    # ── 单图模式 ──────────────────────────────────────────────────
    else:
        predictor.predict(
            image_path  = IMAGE_PATH,
            out_dir     = OUT_DIR,
            city_name   = CITY_NAME,
            min_area_px = MIN_AREA,
            visualize   = VISUALIZE,
        )


if __name__ == '__main__':
    main()