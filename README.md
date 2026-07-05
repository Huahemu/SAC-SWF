# SAC-SWF

**Structure-Aware Combined Side-Window Filtering**

基于结构张量的自适应组合边窗滤波，用于数字图像的结构–纹理分离。

## 方法简介

SAC-SWF 结合三条路径：

1. **结构张量路径**：从梯度计算结构张量，提取特征值得到结构置信度 $R$，衡量当前像素属于强结构的程度。
2. **纹理指示路径**：通过高频残差和振荡项 $O$，衡量当前像素属于纹理/噪声的程度。
3. **边窗候选路径**：生成 full-window 候选 $Q^F$ 和 8 方向 side-window 候选，经软选择聚合为 $Q^S$。

最终输出由组合权重 $\beta = \text{clip}(1 - R + \gamma O)$ 自适应加权：

$$\hat{S} = \beta \cdot Q^F + (1 - \beta) \cdot Q^S$$

在结构边缘处 $\beta \to 0$，选用 side-window 结果保持边缘；在纹理/平坦区 $\beta \to 1$，选用 full-window 结果抑制纹理。

## 目录结构

```
src/sac_swf/          核心算法
  sac_swf.py            主函数：compute_beta() + sac_swf()
  structure_tensor.py   结构张量计算（Sobel 梯度 + 闭式特征值）
  side_window.py        8 方向边窗核生成与聚合
  filters.py            核化引导滤波
  baselines.py          对照方法（BOX/GF/BF/SWGF）
  experiment.py         实验调度框架
  metrics.py            评估指标
  visualization.py      可视化工具
  synthetic.py          合成数据生成
  ...

scripts/               实验脚本（exp01–exp07）
configs/               参数配置
tests/                 冒烟测试
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## 使用

```bash
# 环境检查
python scripts/00_check_environment.py

# 生成合成数据
python scripts/01_generate_synthetic.py --max-samples 12

# 逐个运行实验
python scripts/02_run_window_principle.py
python scripts/03_run_synthetic_experiment.py --limit 6
python scripts/04_run_real_images.py --limit 4
python scripts/05_run_ablation.py --limit 6

# 一键运行全部
python scripts/13_run_full_pipeline.py

# 冒烟测试
python scripts/run_all_smoke.py
```

## 参考文献

- H. Yin et al., "Side-Window Filtering," CVPR 2019.
- K. He et al., "Guided Image Filtering," PAMI 2013.
- C. Tomasi and R. Manduchi, "Bilateral Filtering for Gray and Color Images," ICCV 1998.
