"""
足球场遥感检测系统 - 评估脚本（Kaggle 版）
在测试集上输出：
  - 混淆矩阵热力图
  - Precision-Recall 曲线与 AP 值
  - 各类别指标条形图
  - 随机样本可视化（原始 / 真值 / 预测 / FP-FN 四联图）
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')   # Kaggle 无显示器，必须用非交互后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
from model import ImprovedUNet, FootballFieldDataset


class Evaluator:

    def __init__(self, checkpoint_path=None):
        self.device = torch.device(config.DEVICE)
        Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        ckpt_path = checkpoint_path or config.CHECKPOINT_PATH
        self._load_model(ckpt_path)

    # ──────────────────────────────────────────────────────
    def _load_model(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f'模型不存在: {path}  请先运行 train.py')
        ckpt     = torch.load(path, map_location=self.device)
        cfg      = ckpt.get('config', {})
        in_ch    = cfg.get('input_channels', config.INPUT_CHANNELS)
        n_cls    = cfg.get('num_classes',    config.NUM_CLASSES)
        self.model = ImprovedUNet(in_ch, n_cls).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f'  ✓ 模型加载成功  (训练最佳 IoU={ckpt.get("iou",0):.4f},'
              f' Epoch={ckpt.get("epoch","?")})')

    # ──────────────────────────────────────────────────────
    def evaluate(self):
        """在测试集（或退而用验证集）上计算完整指标"""
        print('\n' + '='*60)
        print('  测试集评估')
        print('='*60)

        try:
            ds = FootballFieldDataset(config.DATASET_DIR, 'test')
            tag = 'test'
        except FileNotFoundError:
            ds = FootballFieldDataset(config.DATASET_DIR, 'val')
            tag = 'val (test 不存在，用 val 代替)'

        loader = DataLoader(ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=2)
        print(f'  评估集 ({tag}): {len(ds)} 个样本')

        n_cls    = config.NUM_CLASSES
        conf_mat = np.zeros((n_cls, n_cls), dtype=np.int64)
        total_loss = 0.0
        probs_list, tgts_list = [], []

        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc='  推理', ncols=80):
                imgs = imgs.to(self.device, non_blocking=True)
                lbls = lbls.to(self.device, non_blocking=True)
                with autocast(enabled=config.USE_AMP):
                    logits = self.model(imgs)
                    loss   = F.cross_entropy(logits, lbls)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                total_loss += loss.item()

                for t, p in zip(lbls.cpu().numpy().ravel(),
                                preds.cpu().numpy().ravel()):
                    conf_mat[t][p] += 1

                # 前景概率用于 PR 曲线
                probs_list.append(probs[:, 1].cpu().numpy().ravel())
                tgts_list.append((lbls == 1).cpu().numpy().ravel().astype(np.uint8))

        metrics = self._calc_metrics(conf_mat)
        metrics['loss'] = total_loss / len(loader)
        self._print_report(metrics, conf_mat)
        self._save_json(metrics, conf_mat)

        all_probs = np.concatenate(probs_list)
        all_tgts  = np.concatenate(tgts_list)
        self._plot(metrics, conf_mat, all_probs, all_tgts)
        return metrics

    # ──────────────────────────────────────────────────────
    def _calc_metrics(self, cm):
        n   = cm.shape[0]
        res = {'per_class': {}, 'overall': {}, 'foreground': {}}
        ious, ps, rs, f1s = [], [], [], []
        for c in range(n):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            iou = tp / (tp+fp+fn+1e-7)
            p   = tp / (tp+fp+1e-7)
            r   = tp / (tp+fn+1e-7)
            f1  = 2*p*r / (p+r+1e-7)
            name = config.CLASS_NAMES[c] if c < len(config.CLASS_NAMES) else f'cls{c}'
            res['per_class'][name] = {
                'iou': float(iou), 'precision': float(p),
                'recall': float(r), 'f1': float(f1),
                'support': int(cm[c, :].sum()),
            }
            ious.append(iou); ps.append(p); rs.append(r); f1s.append(f1)
        res['overall']['mIoU'] = float(np.mean(ious))
        res['overall']['OA']   = float(cm.diagonal().sum() / cm.sum())
        res['foreground'] = {
            'iou':       float(np.mean(ious[1:])),
            'precision': float(np.mean(ps[1:])),
            'recall':    float(np.mean(rs[1:])),
            'f1':        float(np.mean(f1s[1:])),
        }
        return res

    # ──────────────────────────────────────────────────────
    def _print_report(self, metrics, cm):
        print(f'\n  {"类别":10s}  {"IoU":>6}  {"Prec":>6}  {"Recall":>6}  {"F1":>6}  {"支持数":>8}')
        print('  ' + '-'*55)
        for name, m in metrics['per_class'].items():
            print(f'  {name:10s}  {m["iou"]:6.4f}  {m["precision"]:6.4f}'
                  f'  {m["recall"]:6.4f}  {m["f1"]:6.4f}  {m["support"]:8,}')
        fg = metrics['foreground'];  ov = metrics['overall']
        print(f'\n  前景 IoU       : {fg["iou"]:.4f}')
        print(f'  前景 Precision : {fg["precision"]:.4f}')
        print(f'  前景 Recall    : {fg["recall"]:.4f}')
        print(f'  前景 F1        : {fg["f1"]:.4f}')
        print(f'  mIoU（含背景） : {ov["mIoU"]:.4f}')
        print(f'  OA（像素准确率）: {ov["OA"]:.4f}')
        print(f'  Loss           : {metrics.get("loss",0):.4f}')
        print('\n  混淆矩阵（行=真实, 列=预测）:')
        n = cm.shape[0]
        names = config.CLASS_NAMES[:n]
        print(f'  {"":12}', end='')
        for nm in names: print(f'  {nm[:6]:>8}', end='')
        print()
        for i, nm in enumerate(names):
            print(f'  {nm[:12]:12}', end='')
            for j in range(n): print(f'  {cm[i,j]:>8,}', end='')
            print()

    # ──────────────────────────────────────────────────────
    def _save_json(self, metrics, cm):
        data = {'metrics': metrics, 'confusion_matrix': cm.tolist(),
                'config': {'mode': config.CLASSIFICATION_MODE,
                           'num_classes': config.NUM_CLASSES,
                           'class_names': config.CLASS_NAMES}}
        path = Path(config.RESULTS_DIR) / 'evaluation_results.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f'\n  评估结果已保存: {path}')

    # ──────────────────────────────────────────────────────
    def _plot(self, metrics, cm, probs, tgts):
        """3 列图：混淆矩阵 / PR 曲线 / 指标条形图"""
        n    = cm.shape[0]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('足球场检测 - 测试集评估结果', fontsize=14, fontweight='bold')

        # 1. 归一化混淆矩阵
        ax   = axes[0]
        ncm  = cm.astype(float)
        rs   = ncm.sum(1, keepdims=True)
        ncm  = np.divide(ncm, rs, where=rs>0)
        im   = ax.imshow(ncm, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046)
        names = config.CLASS_NAMES[:n]
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha='right')
        ax.set_yticks(range(n)); ax.set_yticklabels(names)
        ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别')
        ax.set_title('混淆矩阵（行归一化）')
        for i in range(n):
            for j in range(n):
                c = 'white' if ncm[i,j] > 0.5 else 'black'
                ax.text(j, i, f'{ncm[i,j]:.2f}\n({cm[i,j]:,})',
                        ha='center', va='center', color=c, fontsize=8)

        # 2. PR 曲线
        ax = axes[1]
        try:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            prec, rec, _ = precision_recall_curve(tgts, probs)
            ap = average_precision_score(tgts, probs)
            ax.plot(rec, prec, 'b-', lw=2, label=f'AP = {ap:.4f}')
            ax.fill_between(rec, prec, alpha=0.1, color='blue')
        except Exception:
            ax.text(0.5, 0.5, 'sklearn 未安装\n无法绘制 PR 曲线',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Recall（召回率）'); ax.set_ylabel('Precision（精确率）')
        ax.set_title('Precision-Recall 曲线（前景类）')
        ax.legend(loc='lower left'); ax.grid(True, alpha=0.3)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.05])

        # 3. 指标条形图
        ax     = axes[2]
        mnms   = ['IoU', 'Precision', 'Recall', 'F1']
        x      = np.arange(len(mnms))
        width  = 0.8 / n
        colors = plt.cm.Set2(np.linspace(0, 1, n))
        for idx, (nm, m) in enumerate(metrics['per_class'].items()):
            vals = [m['iou'], m['precision'], m['recall'], m['f1']]
            ax.bar(x + idx*width, vals, width, label=nm, color=colors[idx], alpha=0.85)
        ax.set_xticks(x + width*(n-1)/2)
        ax.set_xticklabels(mnms); ax.set_ylim([0, 1.1])
        ax.set_title('各类别指标对比')
        ax.legend(loc='lower right'); ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        sp = Path(config.RESULTS_DIR) / 'evaluation_charts.png'
        plt.savefig(sp, dpi=150, bbox_inches='tight'); plt.close()
        print(f'  图表已保存: {sp}')

    # ──────────────────────────────────────────────────────
    def visualize_samples(self, n=8):
        """抽样可视化：原始 RGB / 真值 / 预测 / FP-FN 四联图"""
        print('\n生成样本可视化...')
        try:
            ds = FootballFieldDataset(config.DATASET_DIR, 'test')
        except FileNotFoundError:
            ds = FootballFieldDataset(config.DATASET_DIR, 'val')

        n    = min(n, len(ds))
        idxs = np.random.choice(len(ds), n, replace=False)
        fig, axes = plt.subplots(n, 4, figsize=(13, n*3.2))
        if n == 1: axes = axes[np.newaxis, :]

        for j, t in enumerate(['原始影像 (RGB)', '真值标签', '预测结果', 'FP(红)/FN(绿)']):
            axes[0, j].set_title(t, fontsize=10, pad=6)

        cmap_cls = plt.cm.get_cmap('tab10', config.NUM_CLASSES)

        with torch.no_grad():
            for row, idx in enumerate(idxs):
                img_t, lbl_t = ds[idx]
                with autocast(enabled=config.USE_AMP):
                    logits = self.model(img_t.unsqueeze(0).to(self.device))
                pred = logits.argmax(1).squeeze().cpu().numpy()
                lbl  = lbl_t.numpy()

                # RGB 拉伸
                rgb = img_t.numpy()[:3].transpose(1, 2, 0)
                lo, hi = np.percentile(rgb, [2, 98])
                rgb = np.clip((rgb-lo)/(hi-lo+1e-8), 0, 1)

                # 错误图
                err = np.zeros_like(lbl, dtype=np.uint8)
                err[(pred==1)&(lbl==0)] = 1   # FP 红
                err[(pred==0)&(lbl==1)] = 2   # FN 绿

                for j, (data, kw) in enumerate([
                    (rgb,  {}),
                    (lbl,  dict(cmap=cmap_cls, vmin=0, vmax=config.NUM_CLASSES-1)),
                    (pred, dict(cmap=cmap_cls, vmin=0, vmax=config.NUM_CLASSES-1)),
                    (err,  dict(cmap=plt.cm.get_cmap('RdYlGn_r', 3), vmin=0, vmax=2)),
                ]):
                    axes[row, j].imshow(data, **kw)
                    axes[row, j].axis('off')

        patches = [
            mpatches.Patch(color='#e74c3c', alpha=0.8, label='FP 误检'),
            mpatches.Patch(color='#2ecc71', alpha=0.8, label='FN 漏检'),
        ]
        fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=9)
        plt.suptitle('测试集随机样本可视化', fontsize=12, fontweight='bold')
        plt.tight_layout()
        sp = Path(config.RESULTS_DIR) / 'sample_predictions.png'
        plt.savefig(sp, dpi=150, bbox_inches='tight'); plt.close()
        print(f'  样本图已保存: {sp}')


if __name__ == '__main__':
    ev = Evaluator()
    ev.evaluate()
    ev.visualize_samples(8)
    print('\n评估完成！')
