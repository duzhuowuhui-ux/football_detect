"""
足球场遥感检测系统 - 一键运行脚本（Kaggle 版）

Kaggle 使用方式：
  在 Notebook 最后一个 cell 执行：
      !cd /kaggle/input/football-detection-code/code && python run.py
  或直接按步骤调用各模块（推荐 Notebook 中逐 cell 运行）。

本地使用方式：
  python run.py         → 交互式菜单
  python run.py --all   → 直接执行完整流程（跳过交互）
"""

import sys
import os
import json
from pathlib import Path


def _banner():
    print("""
╔══════════════════════════════════════════════════════╗
║       足球场遥感深度学习检测系统         ║
║  U-Net + ASPP + Attention Gate + Focal/Dice Loss     ║
╚══════════════════════════════════════════════════════╝
""")


def _menu():
    print('─' * 55)
    print('  完整流程')
    print('    1. 数据准备（原始影像 → npy 切片）')
    print('    2. 模型训练')
    print('    3. 模型评估（测试集）')
    print('    4. 预测（全幅影像）')
    print('    5. 完整流程  1→2→3→4')
    print('─' * 55)
    print('  工具')
    print('    6. 打印当前配置')
    print('    7. 绘制训练曲线')
    print('    0. 退出')
    print('─' * 55)


# ============================================================
# 各步骤
# ============================================================

def step_check():
    import config
    ok = True
    print('\n当前配置:')
    print(f'  运行环境   : {"Kaggle" if config.IN_KAGGLE else "本地"}')
    if config.DEVICE == 'cuda':
        import torch
        print(f'  GPU        : {torch.cuda.get_device_name(0)}')
    else:
        print('  设备       : CPU（无 GPU 加速）')
    print(f'  分类模式   : {config.CLASSIFICATION_MODE}  ({config.NUM_CLASSES} 类)')
    print(f'  数据集目录 : {config.DATASET_DIR}')
    ds_ok = Path(config.DATASET_DIR, 'train', 'images').is_dir()
    print(f'    → 数据集存在: {ds_ok}')
    if not ds_ok:
        print('    ⚠ 请先运行步骤 1（数据准备）或上传预处理好的数据集')
        ok = False
    print(f'  模型目录   : {config.MODEL_DIR}')
    ckpt_ok = Path(config.CHECKPOINT_PATH).exists()
    print(f'  模型文件   : {config.CHECKPOINT_PATH}  (存在: {ckpt_ok})')
    print(f'  Batch Size : {config.BATCH_SIZE}')
    print(f'  AMP        : {config.USE_AMP}')
    return ok


def step_prepare():
    import config
    print('\n' + '='*55)
    print('  步骤 1：数据准备')
    print('='*55)

    # 若输入目录已有数据集，直接跳过
    if config.DATASET_DIR.startswith('/kaggle/input'):
        print(f'  ✓ 检测到输入目录中的预处理数据集:')
        print(f'    {config.DATASET_DIR}')
        print('  直接使用，无需重新生成。')
        return True

    from prepare_data import DatasetPreparation
    DatasetPreparation().prepare()
    return True


def step_train():
    import config
    print('\n' + '='*55)
    print('  步骤 2：模型训练')
    print('='*55)
    ds_ok = Path(config.DATASET_DIR, 'train', 'images').is_dir()
    if not ds_ok:
        print(f'  ✗ 数据集不存在: {config.DATASET_DIR}')
        print('    请先运行步骤 1')
        return False
    from train import Trainer
    Trainer().train()
    return True


def step_evaluate():
    import config
    print('\n' + '='*55)
    print('  步骤 3：模型评估')
    print('='*55)
    if not Path(config.CHECKPOINT_PATH).exists():
        print(f'  ✗ 模型不存在: {config.CHECKPOINT_PATH}')
        print('    请先运行步骤 2')
        return False
    from evaluate import Evaluator
    ev = Evaluator()
    ev.evaluate()
    ev.visualize_samples(8)
    return True


def step_predict():
    import config
    print('\n' + '='*55)
    print('  步骤 4：影像预测')
    print('='*55)
    if not Path(config.CHECKPOINT_PATH).exists():
        print(f'  ✗ 模型不存在: {config.CHECKPOINT_PATH}')
        print('    请先运行步骤 2')
        return False
    from predict import Predictor
    Predictor().predict()
    return True


def step_plot():
    import config, json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    hist_path = Path(config.MODEL_DIR) / 'history.json'
    if not hist_path.exists():
        print(f'  ✗ 未找到训练历史: {hist_path}')
        return

    with open(hist_path) as f:
        hist = json.load(f)

    eps = range(1, len(hist['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('训练历史', fontsize=14)

    # 损失
    ax = axes[0]
    ax.plot(eps, hist['train_loss'], label='训练损失', color='steelblue')
    ax.plot(eps, hist['val_loss'],   label='验证损失', color='tomato')
    ax.set_title('Focal + Dice 损失'); ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(alpha=0.3)

    # 指标
    ax = axes[1]
    ax.plot(eps, hist['val_iou'],       label='前景 IoU',  color='green',  lw=2)
    ax.plot(eps, hist['val_precision'], label='Precision', color='orange', lw=1.5, ls='--')
    ax.plot(eps, hist['val_recall'],    label='Recall',    color='purple', lw=1.5, ls='--')
    ax.plot(eps, hist['val_f1'],        label='F1',        color='gray',   lw=1.5, ls=':')
    ax.set_title('验证指标'); ax.set_xlabel('Epoch')
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)

    # 学习率
    ax = axes[2]
    ax.semilogy(eps, hist['lr'], color='gray')
    ax.set_title('学习率（Warmup + 余弦退火）'); ax.set_xlabel('Epoch')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    sp = Path(config.MODEL_DIR) / 'training_curves.png'
    plt.savefig(sp, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  ✓ 训练曲线已保存: {sp}')
    best = max(hist['val_iou'])
    ep   = hist['val_iou'].index(best) + 1
    print(f'  最佳 IoU: {best:.4f}  (Epoch {ep})')


def full_pipeline():
    _banner()
    step_check()
    for name, fn in [('数据准备', step_prepare),
                     ('模型训练', step_train),
                     ('模型评估', step_evaluate),
                     ('影像预测', step_predict)]:
        print(f'\n{"─"*55}  →  {name}')
        ok = fn()
        if not ok:
            print(f'\n  ✗ {name}失败，中止流程。')
            return
    print('\n' + '='*55)
    print('  ✓ 完整流程执行完毕！')
    import config
    print(f'  结果目录: {config.RESULTS_DIR}')
    print('  （Kaggle Output 标签页可查看和下载）')
    print('='*55)


# ============================================================
# 入口
# ============================================================

def main():
    # 命令行 --all 直接跑完整流程（适合 Kaggle cell）
    if '--all' in sys.argv:
        full_pipeline()
        return

    _banner()
    while True:
        _menu()
        choice = input('  请输入选项: ').strip()
        if   choice == '1': step_prepare()
        elif choice == '2': step_train()
        elif choice == '3': step_evaluate()
        elif choice == '4': step_predict()
        elif choice == '5': full_pipeline()
        elif choice == '6': step_check()
        elif choice == '7': step_plot()
        elif choice == '0':
            print('\n  再见！\n'); break
        else:
            print('  无效选项，请重新输入')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n\n  用户中断，程序退出。')
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
