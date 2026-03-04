"""
足球场遥感检测系统 - 配置文件（AutoDL 版）

AutoDL 目录约定：
  数据目录 (快速SSD): /root/autodl-tmp/
  代码目录          : /root/football-detection/   ← 本项目放这里
  输出目录          : /root/autodl-tmp/output/    （自动创建）

上传数据到 AutoDL 的推荐方式：
  1. 使用 AutoDL 网盘 / JupyterLab 文件管理器上传
  2. 或 scp 命令: scp -P <port> ortho_tif.tif root@<host>:/root/autodl-tmp/data/
"""

import os
import torch

# ==================== 环境检测 ====================
IN_KAGGLE  = os.path.exists('/kaggle/input')
IN_AUTODL  = os.path.exists('/root/autodl-tmp')   # AutoDL 标志目录

# ==================== 路径配置 ====================
if IN_KAGGLE:
    # ── Kaggle 路径（保持兼容）────────────────────────────
    _DATA_ROOT           = '/kaggle/input/football-detection/file'
    IMAGE_PATH           = f'{_DATA_ROOT}/ortho_tif.tif'
    ANNOTATION_SHAPEFILE = f'{_DATA_ROOT}/annotations/fields.shp'
    _READY_DATASET       = f'{_DATA_ROOT}/dataset'
    _WORK                = '/kaggle/working'

elif IN_AUTODL:
    # ── AutoDL 路径 ──────────────────────────────────────
    # /root/autodl-tmp 是高速 SSD，推荐存放数据集和模型
    _DATA_ROOT           = 'r/root/autodl-tmp/data'

    # ★ 原始遥感影像与标注文件，上传后填写正确路径
    IMAGE_PATH           = r'/root/autodl-tmp/ortho_tif.tif'
    ANNOTATION_SHAPEFILE = r'/root/autodl-tmp/shp/field1.shp'

    # 预处理好的 npy 数据集目录（prepare_data.py 生成 / 手动上传均可）
    _READY_DATASET       = f'{_DATA_ROOT}/dataset'

    # 输出目录（模型 / 结果 / 日志）
    _WORK                = '/root/autodl-tmp/output'

else:
    # ── 本地路径（按实际情况修改）────────────────────────
    IMAGE_PATH           = r'E:\HK\毕业设计\Data\Wuhan\ortho_tif.tif'
    ANNOTATION_SHAPEFILE = r'E:\QGISprogram\shp\field1.shp'
    _READY_DATASET       = None
    _WORK                = '.'

# ==================== 分类模式 ====================
# 'binary'     → 只检测足球场（推荐）
# 'multiclass' → 区分天然草坪 / 人工草坪
CLASSIFICATION_MODE = 'binary'

if CLASSIFICATION_MODE == 'binary':
    NUM_CLASSES = 2
    CLASS_NAMES = ['背景', '足球场']
    CLASS_FIELD = 'class'
else:
    NUM_CLASSES = 3
    CLASS_NAMES = ['背景', '天然草坪', '人工草坪']
    CLASS_FIELD = 'class'

# ==================== 目录解析 ====================
def _pick_dataset_dir():
    """
    按优先级确定数据集目录：
      1. 预处理好的 npy（_READY_DATASET）—— 最快
      2. 工作目录中上次生成的 npy
      3. 工作目录（待 prepare_data.py 生成）
    """
    candidates = []
    if _READY_DATASET:
        candidates += [
            f'{_READY_DATASET}_{CLASSIFICATION_MODE}',
            _READY_DATASET,
        ]
    candidates += [f'{_WORK}/dataset_{CLASSIFICATION_MODE}']

    for p in candidates:
        train_img = os.path.join(p, 'train', 'images')
        if os.path.isdir(train_img) and os.listdir(train_img):
            return p

    return f'{_WORK}/dataset_{CLASSIFICATION_MODE}'   # 待创建

DATASET_DIR     = _pick_dataset_dir()
MODEL_DIR       = f'{_WORK}/models_{CLASSIFICATION_MODE}'
RESULTS_DIR     = f'{_WORK}/results_{CLASSIFICATION_MODE}'
CHECKPOINT_PATH = f'{MODEL_DIR}/best_model.pth'

# ==================== 数据集划分 ====================
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ==================== 切片参数 ====================
TILE_SIZE = 256        # 显存不足可改 128
OVERLAP   = 64

# ==================== 影像属性 ====================
INPUT_CHANNELS = 4     # RGBN 四波段
MAX_VALUE      = 2047.0   # 11 位影像归一化分母

# ==================== 训练超参数 ====================
# AutoDL RTX 3090 / A100 显存 24GB+，batch=16 合适；A10 可改 8
BATCH_SIZE     = 16 if torch.cuda.is_available() else 2
EPOCHS         = 100
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4
USE_AMP        = True
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS  = 5

# ==================== 损失函数参数 ====================
FOCAL_ALPHA  = 0.75
FOCAL_GAMMA  = 2.0
FOCAL_WEIGHT = 0.5
DICE_WEIGHT  = 0.5

# ==================== 模型架构 ====================
ASPP_DILATIONS     = [1, 6, 12, 18]
USE_ATTENTION_GATE = True

# ==================== 数据增强（论文 Table 4.3 优化参数）====================
USE_AUGMENTATION = True
ZOOM_OUT_FACTOR  = 2.0
ZOOM_IN_FACTOR   = 0.5
AUG_FLIP_PROB    = 0.5
AUG_ROTATE_PROB  = 0.5
AUG_ZOOM_PROB    = 0.30

# ==================== 设备 ====================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== 打印摘要 ====================
if __name__ == '__main__':
    print('=' * 60)
    print('  足球场遥感检测系统 - 配置摘要（AutoDL 版）')
    print('=' * 60)
    if IN_AUTODL:
        env = 'AutoDL'
    elif IN_KAGGLE:
        env = 'Kaggle'
    else:
        env = '本地'
    gpu = torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'
    print(f'  运行环境   : {env}')
    print(f'  计算设备   : {DEVICE}  ({gpu})')
    print(f'  分类模式   : {CLASSIFICATION_MODE}  ({NUM_CLASSES} 类: {CLASS_NAMES})')
    print(f'  数据集     : {DATASET_DIR}')
    print(f'    (目录存在 : {os.path.isdir(DATASET_DIR)})')
    print(f'  模型目录   : {MODEL_DIR}')
    print(f'  结果目录   : {RESULTS_DIR}')
    print('-' * 60)
    print(f'  Batch Size : {BATCH_SIZE}')
    print(f'  Epochs     : {EPOCHS}')
    print(f'  AMP        : {USE_AMP}')
    print(f'  Attn Gate  : {USE_ATTENTION_GATE}')
    print(f'  ASPP 空洞率: {ASPP_DILATIONS}')
    print(f'  Focal α/γ  : {FOCAL_ALPHA}/{FOCAL_GAMMA}')
    print(f'  Zoom-out   : {ZOOM_OUT_FACTOR}x')
    print(f'  Zoom-in    : {ZOOM_IN_FACTOR}x')
    print('=' * 60)
