"""
足球场遥感检测系统 - 数据准备（Kaggle 版）

说明：
  Kaggle 上 /kaggle/input/football-detection/file/dataset/ 目录
  已包含预处理好的 train/val/test npy 文件。
  config.py 的 _pick_dataset_dir() 会优先使用该目录，
  因此通常 无需 运行本脚本。

  只有当输入目录中没有预处理数据，需要从原始
  ortho_tif.tif + shapefile 重新生成时，才运行本脚本。
  输出写入 /kaggle/working/dataset_binary/（可写目录）。
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import config


class DatasetPreparation:

    def __init__(self):
        # Kaggle 输入目录只读，输出必须写到 working
        self.output_dir = Path(f'{config._WORK}/dataset_{config.CLASSIFICATION_MODE}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f'分类模式 : {config.CLASSIFICATION_MODE}')
        print(f'输出目录 : {self.output_dir}')

    # ──────────────────────────────────────────────────────
    def load_image(self):
        try:
            import rasterio
        except ImportError:
            sys.exit('请先安装 rasterio: !pip install rasterio -q')

        print('\n[1/5] 加载遥感影像...')
        if not Path(config.IMAGE_PATH).exists():
            sys.exit(f'影像文件不存在: {config.IMAGE_PATH}')

        with rasterio.open(config.IMAGE_PATH) as src:
            bands = [src.read(i) for i in range(1, 5)]
            image = np.stack(bands, axis=-1).astype(np.float32)   # (H,W,4)
            self.profile   = src.profile
            self.transform = src.transform
            self.img_shape = (src.height, src.width)

        image = np.clip(image / config.MAX_VALUE, 0.0, 1.0)
        print(f'  影像尺寸: {image.shape}')
        return image

    # ──────────────────────────────────────────────────────
    def load_annotation(self):
        try:
            import geopandas as gpd
            import rasterio
            from rasterio.features import rasterize
        except ImportError:
            sys.exit('请先安装 geopandas rasterio: !pip install geopandas rasterio -q')

        print('\n[2/5] 加载 QGIS 标注...')
        if not Path(config.ANNOTATION_SHAPEFILE).exists():
            sys.exit(f'标注文件不存在: {config.ANNOTATION_SHAPEFILE}')

        gdf = gpd.read_file(config.ANNOTATION_SHAPEFILE)
        print(f'  多边形数量: {len(gdf)}')

        with rasterio.open(config.IMAGE_PATH) as src:
            img_crs = src.crs

        if gdf.crs != img_crs:
            gdf = gdf.to_crs(img_crs)

        if config.CLASS_FIELD not in gdf.columns:
            raise ValueError(
                f"shapefile 缺少 '{config.CLASS_FIELD}' 字段，"
                f"可用字段: {list(gdf.columns)}"
            )

        label = np.zeros(self.img_shape, dtype=np.uint8)
        for _, row in gdf.iterrows():
            mask = rasterize(
                [(row.geometry, int(row[config.CLASS_FIELD]))],
                out_shape=self.img_shape,
                transform=self.transform,
                fill=0, dtype=np.uint8,
            )
            label = np.maximum(label, mask)

        unique, counts = np.unique(label, return_counts=True)
        for cls, cnt in zip(unique, counts):
            name = config.CLASS_NAMES[cls] if cls < len(config.CLASS_NAMES) else f'cls{cls}'
            print(f'  {name}: {cnt:,} 像素 ({cnt/label.size*100:.2f}%)')
        return label

    # ──────────────────────────────────────────────────────
    def extract_tiles(self, image, label):
        print('\n[3/5] 滑动窗口切片...')
        h, w = image.shape[:2]
        s    = config.TILE_SIZE - config.OVERLAP
        T    = config.TILE_SIZE
        tiles = [
            {'image': image[y:y+T, x:x+T], 'label': label[y:y+T, x:x+T]}
            for y in range(0, h-T+1, s)
            for x in range(0, w-T+1, s)
            if np.sum(label[y:y+T, x:x+T] > 0) >= 100
        ]
        print(f'  有效切片: {len(tiles)} 个')
        if not tiles:
            print('  ⚠ 未提取到切片！请检查影像/标注坐标系是否一致。')
        return tiles

    # ──────────────────────────────────────────────────────
    def augment_tiles(self, tiles):
        """离线几何增强，含论文 Table 4.3 的 Zoom-in / Zoom-out"""
        if not config.USE_AUGMENTATION:
            print('\n[4/5] 跳过增强')
            return tiles

        try:
            import cv2
        except ImportError:
            sys.exit('请先安装 opencv: !pip install opencv-python-headless -q')

        print(f'\n[4/5] 离线增强（原始 {len(tiles)} 个）...')
        augmented = []
        for tile in tqdm(tiles, desc='  增强', ncols=80):
            img, lbl = tile['image'], tile['label']
            augmented.append({'image': img, 'label': lbl})

            # 翻转
            augmented.append({'image': np.fliplr(img).copy(), 'label': np.fliplr(lbl).copy()})
            augmented.append({'image': np.flipud(img).copy(), 'label': np.flipud(lbl).copy()})

            # 旋转
            for k in [1, 2, 3]:
                augmented.append({'image': np.rot90(img, k).copy(),
                                   'label': np.rot90(lbl, k).copy()})

            # Zoom-out：在更大画布随机放置后缩回原尺寸
            # 论文将系数从 4.0 优化为 2.0，防止小目标特征丢失
            H, W = img.shape[:2]
            f    = config.ZOOM_OUT_FACTOR
            ch, cw = int(H*f), int(W*f)
            canvas_i = np.zeros((ch, cw, img.shape[2]), dtype=img.dtype)
            canvas_l = np.zeros((ch, cw), dtype=lbl.dtype)
            y0 = np.random.randint(0, ch-H+1)
            x0 = np.random.randint(0, cw-W+1)
            canvas_i[y0:y0+H, x0:x0+W] = img
            canvas_l[y0:y0+H, x0:x0+W] = lbl
            zo_img = cv2.resize(canvas_i, (W, H), interpolation=cv2.INTER_LINEAR)
            zo_lbl = cv2.resize(canvas_l, (W, H), interpolation=cv2.INTER_NEAREST)
            augmented.append({'image': zo_img, 'label': zo_lbl})

            # Zoom-in：随机裁剪 [min_scale,1] 后放大回原尺寸
            # 论文将最小比例从 0.3 优化为 0.5，保证目标完整性
            scale  = np.random.uniform(config.ZOOM_IN_FACTOR, 1.0)
            crop_h = max(1, int(H*scale))
            crop_w = max(1, int(W*scale))
            y0 = np.random.randint(0, H-crop_h+1)
            x0 = np.random.randint(0, W-crop_w+1)
            zi_img = cv2.resize(img[y0:y0+crop_h, x0:x0+crop_w], (W, H), cv2.INTER_LINEAR)
            zi_lbl = cv2.resize(lbl[y0:y0+crop_h, x0:x0+crop_w], (W, H), cv2.INTER_NEAREST)
            augmented.append({'image': zi_img, 'label': zi_lbl})

        print(f'  增强后: {len(augmented)} 个  ({len(augmented)//len(tiles)}×)')
        return augmented

    # ──────────────────────────────────────────────────────
    def save_dataset(self, tiles):
        print('\n[5/5] 划分并保存...')
        idx = list(range(len(tiles)))
        tr, tmp = train_test_split(idx, test_size=1-config.TRAIN_RATIO, random_state=42)
        vr = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        va, te = train_test_split(tmp, test_size=1-vr, random_state=42)

        for split, idxs in [('train', tr), ('val', va), ('test', te)]:
            idir = self.output_dir / split / 'images'
            ldir = self.output_dir / split / 'labels'
            idir.mkdir(parents=True, exist_ok=True)
            ldir.mkdir(parents=True, exist_ok=True)
            for i, j in enumerate(idxs):
                np.save(idir / f'{i:05d}.npy', tiles[j]['image'])
                np.save(ldir / f'{i:05d}.npy', tiles[j]['label'])
            print(f'  {split:5s}: {len(idxs):>5} 个样本')

        # 保存元数据
        meta = {
            'mode': config.CLASSIFICATION_MODE,
            'num_classes': config.NUM_CLASSES,
            'class_names': config.CLASS_NAMES,
            'tile_size': config.TILE_SIZE,
            'overlap': config.OVERLAP,
            'augmentation': config.USE_AUGMENTATION,
            'zoom_out': config.ZOOM_OUT_FACTOR,
            'zoom_in': config.ZOOM_IN_FACTOR,
            'total': len(tiles),
        }
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f'\n  数据集保存至: {self.output_dir}')

    # ──────────────────────────────────────────────────────
    def prepare(self):
        print('\n' + '='*55)
        print('  数据准备（改进版）')
        print('='*55)

        # 检查是否已有现成数据集（输入目录）
        if config.DATASET_DIR != str(self.output_dir):
            print(f'\n  ✓ 检测到现成数据集: {config.DATASET_DIR}')
            print('  跳过重新生成，如需强制重新生成请修改 config.py')
            return

        image  = self.load_image()
        label  = self.load_annotation()
        tiles  = self.extract_tiles(image, label)
        if not tiles:
            return
        tiles  = self.augment_tiles(tiles)
        self.save_dataset(tiles)
        print('\n  ✓ 数据准备完成，下一步运行 train.py')


if __name__ == '__main__':
    DatasetPreparation().prepare()
