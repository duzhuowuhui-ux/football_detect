"""
足球场遥感检测系统 - 模型定义（Kaggle 版）
架构：ASPP 多尺度瓶颈 + Attention Gate U-Net
  - ASPP     : 并行多个空洞率卷积，同时捕获局部和全局语义
  - AttnGate : 用高层语义为跳跃连接打注意力分，聚焦稀疏目标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import config


# ============================================================
# 数据集
# ============================================================

class FootballFieldDataset(Dataset):
    """足球场语义分割数据集，加载 .npy 格式切片"""

    def __init__(self, data_dir, split='train'):
        self.image_dir = Path(data_dir) / split / 'images'
        self.label_dir = Path(data_dir) / split / 'labels'
        self.files     = sorted(self.image_dir.glob('*.npy'))

        if not self.files:
            raise FileNotFoundError(
                f'未在 {self.image_dir} 找到 .npy 文件。\n'
                '请确认 config.DATASET_DIR 指向正确路径，\n'
                '或先运行 prepare_data.py 生成数据集。'
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.load(self.files[idx])                         # (H,W,4)
        lbl = np.load(self.label_dir / self.files[idx].name)  # (H,W)
        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        lbl = torch.from_numpy(lbl.copy()).long()
        return img, lbl


def get_dataloaders(batch_size=8, num_workers=2):
    pin = (config.DEVICE == 'cuda')
    train_ds = FootballFieldDataset(config.DATASET_DIR, 'train')
    val_ds   = FootballFieldDataset(config.DATASET_DIR, 'val')
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=pin)
    return train_ld, val_ld


# ============================================================
# 基础块
# ============================================================

class DoubleConv(nn.Module):
    """(Conv-BN-ReLU) × 2"""
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),  nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


# ============================================================
# ASPP（多尺度空洞空间金字塔池化）
# ============================================================
# 对应论文 4.3 节多尺度融合思想：
#   并行多个不同空洞率的卷积，单个特征图同时包含
#   局部细节（dilation=1）→ 大范围语义（dilation=18）

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256, dilations=None):
        super().__init__()
        dilations = dilations or config.ASPP_DILATIONS

        # 1×1 卷积（感受野最小）
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        # 多个空洞率分支
        self.atrous = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
            for d in dilations])

        # 全局平均池化分支（图像级全局上下文）
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        # 融合投影：所有分支拼接 → 降维
        n = 1 + len(dilations) + 1
        self.proj = nn.Sequential(
            nn.Conv2d(n * out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        gap  = F.interpolate(self.gap(x), (h, w), mode='bilinear', align_corners=False)
        parts = [self.conv1(x)] + [b(x) for b in self.atrous] + [gap]
        return self.proj(torch.cat(parts, dim=1))


# ============================================================
# Attention Gate
# ============================================================
# 足球场极稀疏，Attention Gate 让解码器自动聚焦目标区域。
# 用高层特征 g（门控信号）给跳跃连接 x 每个像素打 0~1 的权重。

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g  = nn.Sequential(nn.Conv2d(F_g,   F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x  = nn.Sequential(nn.Conv2d(F_l,   F_int, 1), nn.BatchNorm2d(F_int))
        self.psi  = nn.Sequential(nn.Conv2d(F_int, 1,     1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, x1.shape[-2:], mode='bilinear', align_corners=False)
        return x * self.psi(self.relu(g1 + x1))


# ============================================================
# 改进版 U-Net
# ============================================================

class ImprovedUNet(nn.Module):
    """
    改进版 U-Net：ASPP 瓶颈 + Attention Gate 解码器

    编码器: 4 → 64 → 128 → 256 → 512
    瓶颈 : ASPP(512 → 256)
    解码器: 256 → 128 → 64 → 32 → num_classes（每级含 Attention Gate）
    """
    def __init__(self, in_channels=4, num_classes=2):
        super().__init__()
        attn = config.USE_ATTENTION_GATE

        # 编码器
        self.enc1  = DoubleConv(in_channels, 64);  self.pool1 = nn.MaxPool2d(2)
        self.enc2  = DoubleConv(64, 128);           self.pool2 = nn.MaxPool2d(2)
        self.enc3  = DoubleConv(128, 256);          self.pool3 = nn.MaxPool2d(2)
        self.enc4  = DoubleConv(256, 512);          self.pool4 = nn.MaxPool2d(2)

        # ASPP 瓶颈
        self.bottleneck = ASPP(512, 256, config.ASPP_DILATIONS)

        # 解码器
        self.up4   = nn.ConvTranspose2d(256, 256, 2, 2)
        self.attn4 = AttentionGate(256, 512, 256) if attn else None
        self.dec4  = DoubleConv(256+512, 256)

        self.up3   = nn.ConvTranspose2d(256, 128, 2, 2)
        self.attn3 = AttentionGate(128, 256, 128) if attn else None
        self.dec3  = DoubleConv(128+256, 128)

        self.up2   = nn.ConvTranspose2d(128, 64, 2, 2)
        self.attn2 = AttentionGate(64, 128, 64) if attn else None
        self.dec2  = DoubleConv(64+128, 64)

        self.up1   = nn.ConvTranspose2d(64, 32, 2, 2)
        self.attn1 = AttentionGate(32, 64, 32) if attn else None
        self.dec1  = DoubleConv(32+64, 32)

        self.out_conv = nn.Conv2d(32, num_classes, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

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


def get_model(device=None):
    device = device or config.DEVICE
    model  = ImprovedUNet(config.INPUT_CHANNELS, config.NUM_CLASSES).to(device)
    total  = sum(p.numel() for p in model.parameters())
    print(f'  模型参数量: {total:,}')
    return model


if __name__ == '__main__':
    m = get_model()
    x = torch.randn(2, config.INPUT_CHANNELS, 256, 256).to(config.DEVICE)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, config.NUM_CLASSES, 256, 256)
    print(f'  输入 {list(x.shape)} → 输出 {list(y.shape)}  ✓')
