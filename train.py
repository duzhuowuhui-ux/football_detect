"""
足球场遥感检测系统 - 训练脚本（Kaggle 版）
改进点：
  1. FocalDiceLoss   — 处理足球场极稀疏的类别不平衡
  2. 在线增强         — GPU 上 Zoom-in/Zoom-out + 翻转旋转
  3. AMP 混合精度    — Kaggle P100/T4 速度约 2x，显存减半
  4. AdamW + 余弦退火 — 更稳定的优化策略
  5. 梯度裁剪         — 防止稀疏标签场景梯度爆炸
  6. 实时进度保存     — 每 epoch 写 CSV，便于 Kaggle 监控
"""

import os
import math
import json
import random
import csv
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import config
from model import get_model, get_dataloaders


# ============================================================
# 损失函数
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss：FL(p_t) = -α_t · (1-p_t)^γ · log(p_t)
    降低大量易分类背景像素的梯度权重，让网络专注足球场边界等难例。
    α=0.75 强化正类，γ=2.0 经典聚焦强度。
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets, reduction='none')   # (B,H,W)
        p_t   = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = torch.where(targets > 0,
                            torch.full_like(p_t, self.alpha),
                            torch.full_like(p_t, 1.0 - self.alpha))
        return (alpha * (1.0 - p_t) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    """
    Dice Loss：直接优化预测与真值的区域重叠，对稀疏目标天然平衡。
    仅对前景类（跳过背景 class=0）计算。
    """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs  = F.softmax(logits, dim=1)                     # (B,C,H,W)
        n_cls  = probs.shape[1]
        t_oh   = F.one_hot(targets, n_cls).permute(0,3,1,2).float()  # (B,C,H,W)
        losses = []
        for c in range(1, n_cls):
            p     = probs[:, c].reshape(-1)
            t     = t_oh[:,  c].reshape(-1)
            inter = (p * t).sum()
            losses.append(1.0 - (2*inter + self.smooth) / (p.sum() + t.sum() + self.smooth))
        return torch.stack(losses).mean()


class FocalDiceLoss(nn.Module):
    """组合损失 = 0.5×Focal + 0.5×Dice（比例可在 config 中调整）"""
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(config.FOCAL_ALPHA, config.FOCAL_GAMMA)
        self.dice  = DiceLoss()

    def forward(self, logits, targets):
        return (config.FOCAL_WEIGHT * self.focal(logits, targets) +
                config.DICE_WEIGHT  * self.dice(logits, targets))


# ============================================================
# 在线数据增强（GPU 上执行，避免重复传输）
# ============================================================

class OnlineAugmentor:
    """
    在 GPU tensor 上对每个 batch 随机施加增强，
    包含论文 Table 4.3 推荐的 Zoom-in / Zoom-out 参数。
    """
    def __call__(self, imgs, lbls):
        """
        imgs : (B,C,H,W) cuda tensor
        lbls : (B,H,W)   cuda tensor
        """
        out_imgs, out_lbls = [], []
        for img, lbl in zip(imgs, lbls):
            # 水平 / 垂直翻转
            if random.random() < config.AUG_FLIP_PROB:
                img = torch.flip(img, [2]); lbl = torch.flip(lbl, [1])
            if random.random() < config.AUG_FLIP_PROB:
                img = torch.flip(img, [1]); lbl = torch.flip(lbl, [0])

            # 随机 90°/180°/270° 旋转
            if random.random() < config.AUG_ROTATE_PROB:
                k = random.randint(1, 3)
                img = torch.rot90(img, k, [1, 2])
                lbl = torch.rot90(lbl, k, [0, 1])

            # Zoom-out / Zoom-in（各 zoom_p/2 概率）
            r = random.random()
            if r < config.AUG_ZOOM_PROB / 2:
                img, lbl = self._zoom_out(img, lbl)
            elif r < config.AUG_ZOOM_PROB:
                img, lbl = self._zoom_in(img, lbl)

            out_imgs.append(img)
            out_lbls.append(lbl)
        return torch.stack(out_imgs), torch.stack(out_lbls)

    def _zoom_out(self, img, lbl):
        """缩小增强：论文优化系数 2.0（原默认 4.0）"""
        C, H, W = img.shape
        f       = config.ZOOM_OUT_FACTOR
        ch, cw  = int(H*f), int(W*f)
        ci = torch.zeros(C, ch, cw, dtype=img.dtype, device=img.device)
        cl = torch.zeros(ch, cw,    dtype=lbl.dtype, device=lbl.device)
        y0 = random.randint(0, ch-H);  x0 = random.randint(0, cw-W)
        ci[:, y0:y0+H, x0:x0+W] = img
        cl[y0:y0+H, x0:x0+W]    = lbl
        out_i = F.interpolate(ci.unsqueeze(0), (H,W), mode='bilinear', align_corners=False).squeeze(0)
        out_l = F.interpolate(cl.float().unsqueeze(0).unsqueeze(0), (H,W), mode='nearest').squeeze().long()
        return out_i, out_l

    def _zoom_in(self, img, lbl):
        """放大增强：论文优化最小裁剪比例 0.5（原默认 0.3）"""
        C, H, W = img.shape
        s       = random.uniform(config.ZOOM_IN_FACTOR, 1.0)
        ch, cw  = max(1, int(H*s)), max(1, int(W*s))
        y0 = random.randint(0, H-ch);  x0 = random.randint(0, W-cw)
        out_i = F.interpolate(img[:, y0:y0+ch, x0:x0+cw].unsqueeze(0),
                              (H,W), mode='bilinear', align_corners=False).squeeze(0)
        out_l = F.interpolate(lbl[y0:y0+ch, x0:x0+cw].float().unsqueeze(0).unsqueeze(0),
                              (H,W), mode='nearest').squeeze().long()
        return out_i, out_l


# ============================================================
# 学习率调度：线性 Warmup + 余弦退火
# ============================================================

def build_scheduler(optimizer, total_epochs, warmup=5):
    """前 warmup 轮线性升温，之后余弦退火至初始 lr 的 1%"""
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / max(1, warmup)
        p = (ep - warmup) / max(1, total_epochs - warmup)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * p))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# ============================================================
# 训练器
# ============================================================

class Trainer:

    def __init__(self):
        Path(config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.DEVICE)

        print('=' * 60)
        print('  训练配置（Kaggle 版）')
        print('=' * 60)
        if config.DEVICE == 'cuda':
            print(f'  GPU        : {torch.cuda.get_device_name(0)}')
        print(f'  分类模式   : {config.CLASSIFICATION_MODE}  ({config.NUM_CLASSES} 类)')
        print(f'  数据集     : {config.DATASET_DIR}')
        print(f'  模型目录   : {config.MODEL_DIR}')
        print(f'  Batch Size : {config.BATCH_SIZE}')
        print(f'  Epochs     : {config.EPOCHS}')
        print(f'  AMP        : {config.USE_AMP}')
        print(f'  Attn Gate  : {config.USE_ATTENTION_GATE}')
        print(f'  ASPP 空洞率: {config.ASPP_DILATIONS}')
        print('=' * 60)

        self.model   = get_model(self.device)
        self.train_ld, self.val_ld = get_dataloaders(config.BATCH_SIZE)
        print(f'\n  训练集: {len(self.train_ld.dataset):,} 个')
        print(f'  验证集: {len(self.val_ld.dataset):,} 个\n')

        self.criterion = FocalDiceLoss()
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=config.LEARNING_RATE,
                                     weight_decay=config.WEIGHT_DECAY)
        self.scheduler = build_scheduler(self.optimizer, config.EPOCHS, config.WARMUP_EPOCHS)
        self.scaler    = GradScaler(enabled=config.USE_AMP)
        self.augmentor = OnlineAugmentor() if config.USE_AUGMENTATION else None

        self.best_iou = 0.0
        self.history  = {k: [] for k in
                         ['train_loss','val_loss','val_iou',
                          'val_precision','val_recall','val_f1','lr']}

    # ──────────────────────────────────────────────────────
    def _metrics(self, logits, targets):
        """计算前景类（足球场）的 IoU / Precision / Recall / F1"""
        pred = logits.argmax(1)
        ious, ps, rs = [], [], []
        for c in range(1, config.NUM_CLASSES):
            pc = (pred == c); tc = (targets == c)
            tp = (pc & tc).sum().item()
            fp = (pc & ~tc).sum().item()
            fn = (~pc & tc).sum().item()
            ious.append(tp / (tp+fp+fn+1e-7))
            ps.append(tp / (tp+fp+1e-7))
            rs.append(tp / (tp+fn+1e-7))
        iou = float(np.mean(ious));  p = float(np.mean(ps));  r = float(np.mean(rs))
        f1  = 2*p*r / (p+r+1e-7)
        return {'iou': iou, 'precision': p, 'recall': r, 'f1': f1}

    # ──────────────────────────────────────────────────────
    def _train_one_epoch(self, ep):
        self.model.train()
        total = 0.0
        pbar  = tqdm(self.train_ld, desc=f'Ep{ep:3d} train', ncols=90, leave=False)
        for imgs, lbls in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            lbls = lbls.to(self.device, non_blocking=True)
            if self.augmentor:
                imgs, lbls = self.augmentor(imgs, lbls)
            with autocast(enabled=config.USE_AMP):
                loss = self.criterion(self.model(imgs), lbls)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), config.GRAD_CLIP_NORM)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        return total / len(self.train_ld)

    # ──────────────────────────────────────────────────────
    def _validate(self):
        self.model.eval()
        total = 0.0
        all_m = {k: [] for k in ['iou','precision','recall','f1']}
        with torch.no_grad():
            for imgs, lbls in tqdm(self.val_ld, desc='       val  ', ncols=90, leave=False):
                imgs = imgs.to(self.device, non_blocking=True)
                lbls = lbls.to(self.device, non_blocking=True)
                with autocast(enabled=config.USE_AMP):
                    logits = self.model(imgs)
                    loss   = self.criterion(logits, lbls)
                total += loss.item()
                for k, v in self._metrics(logits, lbls).items():
                    all_m[k].append(v)
        return total / len(self.val_ld), {k: float(np.mean(v)) for k,v in all_m.items()}

    # ──────────────────────────────────────────────────────
    def _save_checkpoint(self, ep, m):
        torch.save({
            'epoch': ep,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iou': m['iou'], 'metrics': m, 'history': self.history,
            'config': {
                'num_classes': config.NUM_CLASSES,
                'mode': config.CLASSIFICATION_MODE,
                'use_attention': config.USE_ATTENTION_GATE,
                'aspp_dilations': config.ASPP_DILATIONS,
                'input_channels': config.INPUT_CHANNELS,
            },
        }, config.CHECKPOINT_PATH)

    # ──────────────────────────────────────────────────────
    def train(self):
        # 初始化 CSV 日志（Kaggle 可直接查看）
        csv_path = Path(config.MODEL_DIR) / 'training_log.csv'
        csv_f    = open(csv_path, 'w', newline='')
        writer   = csv.writer(csv_f)
        writer.writerow(['epoch','train_loss','val_loss','val_iou',
                         'val_precision','val_recall','val_f1','lr'])

        print('\n开始训练...\n')
        for ep in range(1, config.EPOCHS + 1):
            tr_loss           = self._train_one_epoch(ep)
            va_loss, va_m     = self._validate()
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']

            # 记录
            for k, v in zip(
                ['train_loss','val_loss','val_iou','val_precision','val_recall','val_f1','lr'],
                [tr_loss, va_loss, va_m['iou'], va_m['precision'], va_m['recall'], va_m['f1'], lr]
            ):
                self.history[k].append(v)

            # 打印（Kaggle 日志中可见）
            print(f'Epoch {ep:3d}/{config.EPOCHS}'
                  f'  loss {tr_loss:.4f}→{va_loss:.4f}'
                  f'  IoU {va_m["iou"]:.4f}'
                  f'  P {va_m["precision"]:.3f}'
                  f'  R {va_m["recall"]:.3f}'
                  f'  F1 {va_m["f1"]:.3f}'
                  f'  lr {lr:.2e}')

            # CSV 实时写入（Kaggle 可监控）
            writer.writerow([ep, tr_loss, va_loss, va_m['iou'],
                             va_m['precision'], va_m['recall'], va_m['f1'], lr])
            csv_f.flush()

            # 保存最佳模型
            if va_m['iou'] > self.best_iou:
                self.best_iou = va_m['iou']
                self._save_checkpoint(ep, va_m)
                print(f'  ✓ 保存最佳模型  IoU={va_m["iou"]:.4f}')

        csv_f.close()

        # 保存完整历史 JSON
        with open(Path(config.MODEL_DIR) / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

        print('\n' + '='*60)
        print(f'  训练完成！最佳前景 IoU : {self.best_iou:.4f}')
        print(f'  模型保存  : {config.CHECKPOINT_PATH}')
        print(f'  训练日志  : {csv_path}')
        print('  下一步    : 运行 evaluate.py 评估，predict.py 预测')
        print('='*60)

        return self.best_iou


if __name__ == '__main__':
    Trainer().train()
