"""
足球场遥感检测系统 - 训练脚本（域泛化版）
==========================================
相比原版新增：辐射增强（RadiometricAugmentor）
  在几何增强的基础上随机扰动亮度、对比度、Gamma、波段权重，
  模拟不同城市、不同传感器、不同季节的辐射差异，
  迫使模型学习与辐射风格无关的结构特征（足球场形状、纹理），
  从而提升跨城市泛化能力。
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
# 损失函数（与原版完全相同）
# ============================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce    = F.cross_entropy(logits, targets, reduction='none')
        p_t   = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        alpha = torch.where(targets > 0,
                            torch.full_like(p_t, self.alpha),
                            torch.full_like(p_t, 1.0 - self.alpha))
        return (alpha * (1.0 - p_t) ** self.gamma * ce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        n_cls = probs.shape[1]
        t_oh  = F.one_hot(targets, n_cls).permute(0,3,1,2).float()
        losses = []
        for c in range(1, n_cls):
            p     = probs[:, c].reshape(-1)
            t     = t_oh[:,  c].reshape(-1)
            inter = (p * t).sum()
            losses.append(1.0 - (2*inter + self.smooth) / (p.sum() + t.sum() + self.smooth))
        return torch.stack(losses).mean()


class FocalDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(config.FOCAL_ALPHA, config.FOCAL_GAMMA)
        self.dice  = DiceLoss()

    def forward(self, logits, targets):
        return (config.FOCAL_WEIGHT * self.focal(logits, targets) +
                config.DICE_WEIGHT  * self.dice(logits, targets))


# ============================================================
# ★ 新增：辐射增强（模拟跨城市传感器差异）
# ============================================================

class RadiometricAugmentor:
    """
    对 GPU tensor 施加随机辐射扰动，模拟不同城市影像的辐射差异。
    只作用于影像，不改变标注掩膜。

    包含四种增强，各自独立以一定概率触发：
    1. 亮度偏移   : 模拟大气透过率、光照强度差异
    2. 对比度缩放 : 模拟传感器动态范围差异
    3. Gamma 校正 : 模拟非线性辐射响应差异
    4. 波段权重   : 模拟不同传感器各波段灵敏度差异

    参考：
      Toker et al. (2022) "DynamicEarthNet" CVPR —— 遥感时序域泛化
      中的辐射一致性分析；
      Maggiori et al. (2017) "Can Semantic Labeling Methods Generalize
      to Any City?" IGARSS —— 跨城市泛化中辐射差异是主要障碍之一。
    """

    def __init__(self,
                 brightness_range=(-0.15, 0.15),   # 亮度偏移范围
                 contrast_range=(0.75, 1.25),       # 对比度缩放范围
                 gamma_range=(0.7, 1.4),            # Gamma 范围（<1 变亮，>1 变暗）
                 band_weight_std=0.08,              # 波段权重扰动标准差
                 p_brightness=0.5,
                 p_contrast=0.5,
                 p_gamma=0.3,
                 p_band=0.3):
        self.brightness_range = brightness_range
        self.contrast_range   = contrast_range
        self.gamma_range      = gamma_range
        self.band_weight_std  = band_weight_std
        self.p_brightness = p_brightness
        self.p_contrast   = p_contrast
        self.p_gamma      = p_gamma
        self.p_band       = p_band

    def __call__(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, C, H, W) GPU tensor，值域已归一化（约 [0, 0.07]）
        返回同形状 tensor，值域保持不变
        """
        B, C, H, W = imgs.shape
        out = imgs.clone()

        for i in range(B):
            img = out[i]   # (C, H, W)

            # 1. 亮度偏移
            if random.random() < self.p_brightness:
                delta = random.uniform(*self.brightness_range)
                img = img + delta

            # 2. 对比度缩放（以均值为中心缩放）
            if random.random() < self.p_contrast:
                factor = random.uniform(*self.contrast_range)
                mean   = img.mean()
                img    = (img - mean) * factor + mean

            # 3. Gamma 校正（仅对正值有效）
            if random.random() < self.p_gamma:
                gamma = random.uniform(*self.gamma_range)
                img   = torch.sign(img) * torch.abs(img).clamp(min=1e-8) ** gamma

            # 4. 波段权重扰动（每个波段乘以接近 1 的随机系数）
            if random.random() < self.p_band:
                weights = 1.0 + torch.randn(C, 1, 1,
                                            device=img.device) * self.band_weight_std
                img = img * weights

            out[i] = img

        # 值域截断（保持与训练输入一致的范围）
        return out.clamp(0.0, 1.0)


# ============================================================
# 在线几何增强（与原版完全相同）
# ============================================================

class OnlineAugmentor:
    def __call__(self, imgs, lbls):
        out_imgs, out_lbls = [], []
        for img, lbl in zip(imgs, lbls):
            if random.random() < config.AUG_FLIP_PROB:
                img = torch.flip(img, [2]); lbl = torch.flip(lbl, [1])
            if random.random() < config.AUG_FLIP_PROB:
                img = torch.flip(img, [1]); lbl = torch.flip(lbl, [0])
            if random.random() < config.AUG_ROTATE_PROB:
                k = random.randint(1, 3)
                img = torch.rot90(img, k, [1, 2])
                lbl = torch.rot90(lbl, k, [0, 1])
            r = random.random()
            if r < config.AUG_ZOOM_PROB / 2:
                img, lbl = self._zoom_out(img, lbl)
            elif r < config.AUG_ZOOM_PROB:
                img, lbl = self._zoom_in(img, lbl)
            out_imgs.append(img)
            out_lbls.append(lbl)
        return torch.stack(out_imgs), torch.stack(out_lbls)

    def _zoom_out(self, img, lbl):
        C, H, W = img.shape
        f       = config.ZOOM_OUT_FACTOR
        ch, cw  = int(H*f), int(W*f)
        ci = torch.zeros(C, ch, cw, dtype=img.dtype, device=img.device)
        cl = torch.zeros(ch, cw,    dtype=lbl.dtype, device=lbl.device)
        y0 = random.randint(0, ch-H); x0 = random.randint(0, cw-W)
        ci[:, y0:y0+H, x0:x0+W] = img
        cl[y0:y0+H, x0:x0+W]    = lbl
        out_i = F.interpolate(ci.unsqueeze(0), (H,W), mode='bilinear', align_corners=False).squeeze(0)
        out_l = F.interpolate(cl.float().unsqueeze(0).unsqueeze(0), (H,W), mode='nearest').squeeze().long()
        return out_i, out_l

    def _zoom_in(self, img, lbl):
        C, H, W = img.shape
        s       = random.uniform(config.ZOOM_IN_FACTOR, 1.0)
        ch, cw  = max(1, int(H*s)), max(1, int(W*s))
        y0 = random.randint(0, H-ch); x0 = random.randint(0, W-cw)
        out_i = F.interpolate(img[:, y0:y0+ch, x0:x0+cw].unsqueeze(0),
                              (H,W), mode='bilinear', align_corners=False).squeeze(0)
        out_l = F.interpolate(lbl[y0:y0+ch, x0:x0+cw].float().unsqueeze(0).unsqueeze(0),
                              (H,W), mode='nearest').squeeze().long()
        return out_i, out_l


# ============================================================
# 学习率调度（与原版完全相同）
# ============================================================

def build_scheduler(optimizer, total_epochs, warmup=5):
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
        print('  训练配置（域泛化版）')
        print('=' * 60)
        if config.DEVICE == 'cuda':
            print(f'  GPU        : {torch.cuda.get_device_name(0)}')
        print(f'  归一化层   : InstanceNorm2d（替代 BatchNorm2d）')
        print(f'  辐射增强   : 启用（模拟跨城市传感器差异）')
        print(f'  分类模式   : {config.CLASSIFICATION_MODE}  ({config.NUM_CLASSES} 类)')
        print(f'  数据集     : {config.DATASET_DIR}')
        print(f'  模型目录   : {config.MODEL_DIR}')
        print(f'  Batch Size : {config.BATCH_SIZE}')
        print(f'  Epochs     : {config.EPOCHS}')
        print(f'  AMP        : {config.USE_AMP}')
        print(f'  Attn Gate  : {config.USE_ATTENTION_GATE}')
        print(f'  ASPP 空洞率: {config.ASPP_DILATIONS}')
        print('=' * 60)

        self.model        = get_model(self.device)
        self.train_ld, self.val_ld = get_dataloaders(config.BATCH_SIZE)
        print(f'\n  训练集: {len(self.train_ld.dataset):,} 个')
        print(f'  验证集: {len(self.val_ld.dataset):,} 个\n')

        self.criterion    = FocalDiceLoss()
        self.optimizer    = optim.AdamW(self.model.parameters(),
                                        lr=config.LEARNING_RATE,
                                        weight_decay=config.WEIGHT_DECAY)
        self.scheduler    = build_scheduler(self.optimizer, config.EPOCHS)
        self.scaler       = GradScaler(enabled=config.USE_AMP)
        self.geo_aug      = OnlineAugmentor()
        self.rad_aug      = RadiometricAugmentor()   # ★ 新增辐射增强
        self.best_iou     = 0.0
        self.history      = []

    # ── 训练一个 epoch ──────────────────────────────────────────
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        with tqdm(self.train_ld, desc=f'Epoch {epoch:3d} train',
                  ncols=90, leave=False) as pbar:
            for imgs, lbls in pbar:
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)

                # 几何增强（与原版相同）
                imgs, lbls = self.geo_aug(imgs, lbls)

                # ★ 辐射增强（仅作用于图像，不改变标注）
                imgs = self.rad_aug(imgs)

                with autocast(enabled=config.USE_AMP):
                    loss = self.criterion(self.model(imgs), lbls)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}')

        return total_loss / len(self.train_ld)

    # ── 验证一个 epoch ──────────────────────────────────────────
    def _val_epoch(self):
        self.model.eval()
        total_loss = 0.0
        ious = []

        with torch.no_grad():
            for imgs, lbls in self.val_ld:
                imgs = imgs.to(self.device)
                lbls = lbls.to(self.device)

                with autocast(enabled=config.USE_AMP):
                    logits = self.model(imgs)
                    loss   = self.criterion(logits, lbls)

                total_loss += loss.item()
                pred = logits.argmax(1)
                for c in range(1, config.NUM_CLASSES):
                    pc = (pred == c); tc = (lbls == c)
                    tp = (pc & tc).sum().item()
                    fp = (pc & ~tc).sum().item()
                    fn = (~pc & tc).sum().item()
                    ious.append(tp / (tp + fp + fn + 1e-7))

        return total_loss / len(self.val_ld), float(np.mean(ious))

    # ── 主训练循环 ──────────────────────────────────────────────
    def train(self):
        log_path = Path(config.MODEL_DIR) / 'training_log.csv'
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_iou', 'lr'])

        for epoch in range(1, config.EPOCHS + 1):
            tr_loss           = self._train_epoch(epoch)
            val_loss, val_iou = self._val_epoch()
            lr                = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            print(f'Ep {epoch:3d}/{config.EPOCHS}  '
                  f'tr={tr_loss:.4f}  val={val_loss:.4f}  '
                  f'IoU={val_iou:.4f}  lr={lr:.1e}')

            with open(log_path, 'a', newline='') as f:
                csv.writer(f).writerow([epoch, tr_loss, val_loss, val_iou, lr])

            self.history.append({'epoch': epoch, 'train_loss': tr_loss,
                                 'val_loss': val_loss, 'val_iou': val_iou})

            if val_iou > self.best_iou:
                self.best_iou = val_iou
                save_path = Path(config.MODEL_DIR) / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'iou':   val_iou,
                    'model_state_dict': self.model.state_dict(),
                    'config': {
                        'input_channels': config.INPUT_CHANNELS,
                        'num_classes':    config.NUM_CLASSES,
                        'use_attention':  config.USE_ATTENTION_GATE,
                        'norm_layer':     'InstanceNorm2d',
                    }
                }, save_path)
                print(f'  ✓ 保存最佳模型  IoU={val_iou:.4f}  → {save_path}')

        json.dump(self.history,
                  open(Path(config.MODEL_DIR)/'history.json','w'), indent=2)
        print(f'\n训练完成  最佳验证 IoU: {self.best_iou:.4f}')


if __name__ == '__main__':
    Trainer().train()