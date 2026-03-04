"""
足球场遥感检测系统 - 预测脚本（Kaggle 版）
对整幅遥感影像进行滑动窗口推理，输出检测掩膜和统计结果。
Kaggle 注意：结果保存至 /kaggle/working/results_*/ （可下载）
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import config
from model import ImprovedUNet


class Predictor:

    def __init__(self, checkpoint_path=None):
        self.device = torch.device(config.DEVICE)
        Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_path or config.CHECKPOINT_PATH
        self._load_model(ckpt_path)

    # ──────────────────────────────────────────────────────
    def _load_model(self, path):
        if not Path(path).exists():
            sys.exit(f'模型不存在: {path}  请先运行 train.py')
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt.get('config', {})
        in_ch = cfg.get('input_channels', config.INPUT_CHANNELS)
        n_cls = cfg.get('num_classes',    config.NUM_CLASSES)
        self.model = ImprovedUNet(in_ch, n_cls).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f'  ✓ 模型加载成功  (IoU={ckpt.get("iou",0):.4f}, Ep={ckpt.get("epoch","?")})')

    # ──────────────────────────────────────────────────────
    def load_image(self, path=None):
        try:
            import rasterio
        except ImportError:
            sys.exit('请先安装 rasterio: !pip install rasterio -q')

        path = path or config.IMAGE_PATH
        if not Path(path).exists():
            sys.exit(f'影像文件不存在: {path}')

        print(f'\n加载影像: {path}')
        with rasterio.open(path) as src:
            bands = [src.read(i) for i in range(1, 5)]
            image = np.stack(bands, axis=-1).astype(np.float32)  # (H,W,4)
            self.profile   = src.profile
            self.transform = src.transform

        image = np.clip(image / config.MAX_VALUE, 0.0, 1.0)
        print(f'  影像尺寸: {image.shape}')
        return image

    # ──────────────────────────────────────────────────────
    def predict_image(self, image):
        """
        滑动窗口预测 + 重叠区域概率平均
        消除切片边界伪影，保证输出连续性。
        """
        print('\n滑动窗口预测...')
        h, w = image.shape[:2]
        T    = config.PRED_TILE_SIZE
        step = T - config.PRED_OVERLAP
        n    = config.NUM_CLASSES

        prob_sum  = np.zeros((h, w, n), dtype=np.float32)
        count_map = np.zeros((h, w),    dtype=np.float32)

        positions = [(x, y)
                     for y in range(0, h-T+1, step)
                     for x in range(0, w-T+1, step)]

        with torch.no_grad():
            for x, y in tqdm(positions, desc='  切片推理', ncols=80):
                tile = image[y:y+T, x:x+T]
                t    = torch.from_numpy(tile).permute(2,0,1).unsqueeze(0).to(self.device)
                with autocast(enabled=config.USE_AMP):
                    logits = self.model(t)
                probs = F.softmax(logits, dim=1).squeeze(0).permute(1,2,0)
                probs = probs.cpu().float().numpy()
                prob_sum[y:y+T, x:x+T]  += probs
                count_map[y:y+T, x:x+T] += 1.0

        count_map = np.maximum(count_map, 1.0)
        prob_sum /= count_map[:, :, np.newaxis]
        prediction = np.argmax(prob_sum, axis=2).astype(np.uint8)
        return prediction, prob_sum

    # ──────────────────────────────────────────────────────
    def _save_tif(self, data, name):
        try:
            import rasterio
        except ImportError:
            return

        profile = self.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
        p = Path(config.RESULTS_DIR) / name
        with rasterio.open(p, 'w', **profile) as dst:
            dst.write(data.astype(np.uint8), 1)
        print(f'  保存: {p}')

    def _save_prob(self, prob_sum):
        try:
            import rasterio
        except ImportError:
            return

        profile = self.profile.copy()
        profile.update(dtype=rasterio.float32,
                       count=config.NUM_CLASSES, compress='lzw')
        p = Path(config.RESULTS_DIR) / 'probability_maps.tif'
        with rasterio.open(p, 'w', **profile) as dst:
            for c in range(config.NUM_CLASSES):
                dst.write(prob_sum[:, :, c].astype(np.float32), c+1)
        print(f'  保存: {p}')

    # ──────────────────────────────────────────────────────
    def _count_fields(self, prediction):
        """连通域分析统计检测到的足球场数量"""
        try:
            import cv2
        except ImportError:
            return {}

        stats = {}
        if config.CLASSIFICATION_MODE == 'binary':
            mask = (prediction == 1).astype(np.uint8) * 255
            self._save_tif(mask, 'football_fields.tif')
            n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
            valid = sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 50)
            stats = {'total': valid}
        else:
            for cls_id, fname in [(1, 'natural_fields.tif'), (2, 'artificial_fields.tif')]:
                mask = (prediction == cls_id).astype(np.uint8) * 255
                self._save_tif(mask, fname)
                n, _, st, _ = cv2.connectedComponentsWithStats(mask, 8)
                cnt = sum(1 for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 50)
                stats[config.CLASS_NAMES[cls_id]] = cnt
            stats['total'] = sum(v for k, v in stats.items() if k != 'total')
        return stats

    # ──────────────────────────────────────────────────────
    def predict(self, image_path=None):
        print('\n' + '='*60)
        print('  开始预测（Kaggle 版）')
        print('='*60)

        image = self.load_image(image_path)
        pred, prob_sum = self.predict_image(image)

        print('\n保存结果...')
        self._save_tif(pred, 'prediction_full.tif')
        self._save_prob(prob_sum)
        stats = self._count_fields(pred)

        print('\n' + '='*60)
        print('  预测完成')
        print('='*60)
        if config.CLASSIFICATION_MODE == 'binary':
            print(f'  检测到足球场 : {stats.get("total", "N/A")} 个')
        else:
            for k, v in stats.items():
                print(f'  {k} : {v} 个')
        print(f'\n  结果目录 : {config.RESULTS_DIR}')
        print('  （在 Kaggle Output 标签页可下载结果文件）')
        return pred, stats


if __name__ == '__main__':
    Predictor().predict()
