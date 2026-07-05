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
src/sac_swf/                核心算法
  __init__.py                 包入口，导出 SACSWFConfig / sac_swf
  sac_swf.py                  主函数：compute_beta() + sac_swf()
  structure_tensor.py         结构张量（Sobel 梯度 + 闭式特征值）
  side_window.py              8 方向边窗核生成与软聚合
  filters.py                  核化引导滤波 / 盒子滤波 / 高斯滤波 / 双边滤波
  baselines.py                对照方法封装（GF / BF / SWGF / hard-SWGF 等）
  utils.py                    图像格式转换工具

scripts/                    报告图表生成脚本
  generate_paper_figures.py   生成 fig01–fig08 + 附录图（机制/候选/权重/平滑/纹理/增强/消融/运行时间）
  generate_extra_figures.py   生成 fig09–fig12（频域响应/参数分析/去噪/棋盘格）
  generate_method_figures.py  生成方法部分示意图（中心vs边窗/窗口定义/β行为）

configs/default.yaml        默认参数配置
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
# 生成报告中的全部实验图（输出到 reports/paper_figures_new/）
python scripts/generate_paper_figures.py
python scripts/generate_extra_figures.py
python scripts/generate_method_figures.py
```

## 参考文献

- H. Yin et al., "Side-Window Filtering," CVPR 2019.
- K. He et al., "Guided Image Filtering," PAMI 2013.
- C. Tomasi and R. Manduchi, "Bilateral Filtering for Gray and Color Images," ICCV 1998.
