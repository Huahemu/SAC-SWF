# SAC-SWF 数字图像处理课程项目

本项目实现 **SAC-SWF: Structure-Aware Combined Side-Window Filtering**，中文可写为“结构感知的组合边窗滤波”。目标任务是结构-纹理分离：在抑制纹理、噪声和细碎振荡的同时，尽量保留物体轮廓、区域边界、角点和细线结构。

请下游执行 agent 严格在本目录运行：`E:\数字图像处理`。不要移动、重命名或删除项目目录，不要把手工挑图当作算法输出。

## 固定目录

```text
E:\数字图像处理
  README.md
  requirements.txt
  pyproject.toml
  configs/
  src/sac_swf/
  scripts/
  data/
    benchmark/
    raw/real/
    raw/eps/
    raw/learning_outputs/
    synthetic/
  results/
  logs/
  reports/figures/
  docs/
  tests/
```

## 安装

```powershell
cd /d E:\数字图像处理
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

如果忘记 `pip install -e .`，多数脚本仍可从项目根目录运行，因为 `scripts/_bootstrap.py` 会临时加入 `src/`。但正式运行仍建议安装 editable 包。

## 实验编号

实验分为两部分。

创新方法实验：

1. `exp01_window_principle`：支撑域/边窗原理实验。
2. `exp02_synthetic`：有真值的合成结构-纹理分离实验。
3. `exp03_real_images`：真实图像视觉分析实验。
4. `exp04_ablation`：SAC-SWF 消融实验。
5. `exp05_benchmark`：Benchmark 实验。优先使用用户放入 `data/raw/eps/` 的官方 EPS 数据；若没有官方 EPS，则使用 `data/benchmark/` 中确定性生成的 course benchmark suite。
6. `exp06_learning_comparison`：学习/数据驱动对比实验。项目内会训练 `patch_ridge` 轻量监督学习基线，并继续支持外部 DnCNN、DRUNet、Restormer 等输出接口。

论文实验复现支撑：

7. `exp07_paper_reproduction`：对双边滤波、导向滤波、SWF/SWGF、固定组合和 SAC-SWF 做 paper-style 控制样例、残差、中间图和 line profile。该实验用于附录支撑，不声称 bit-exact 复现原作者代码。

## 正式运行

推荐一键运行并保存每步日志：

```powershell
python scripts\13_run_full_pipeline.py --synthetic-count 12 --eval-limit 6 --real-limit 4 --repro-size 192
```

日志会保存到：

```text
logs/full_pipeline_YYYYMMDD_HHMMSS/
```

如果要逐步运行：

```powershell
python scripts\00_check_environment.py
python scripts\01_generate_synthetic.py --max-samples 12
python scripts\09_prepare_real_images.py --limit 4
python scripts\02_run_window_principle.py
python scripts\03_run_synthetic_experiment.py --limit 6
python scripts\04_run_real_images.py --limit 4
python scripts\05_run_ablation.py --limit 6
python scripts\14_prepare_benchmark_data.py --real-limit 4 --synthetic-limit 6
python scripts\10_run_exp05_benchmark.py --limit 6
python scripts\11_run_exp06_learning_comparison.py --train-limit 4 --test-limit 4
python scripts\12_run_exp07_paper_reproduction.py --size 192
python scripts\08_collect_tables.py
python scripts\06_make_report_figures.py
python scripts\07_validate_results.py --result results
```

## 实验 6 学习方法

实验 6 默认训练一个项目内轻量数据驱动基线：

```text
patch_ridge: input local patch -> structure_gt center pixel
```

实验 6 还接入了三个作者公开深度网络复现项：

```text
official_dncnn      # cszn/DnCNN + cszn/KAIR dncnn_25.pth
official_drunet     # cszn/DPIR drunet_gray.pth
official_restormer  # swz30/Restormer gaussian_gray_denoising_sigma25.pth
```

先检查官方源码和权重状态：

```powershell
python scripts\16_check_official_deep_assets.py
```

如需下载或续传官方权重：

```powershell
python scripts\17_download_official_deep_weights.py --method official_drunet --max-time 3600
python scripts\17_download_official_deep_weights.py --method official_restormer --max-time 3600
```

输出会保存模型、训练日志、划分和指标：

```text
results/exp06_learning_comparison/learned_models/
results/exp06_learning_comparison/train_test_split.csv
results/exp06_learning_comparison/metrics_all.csv
```

如需加入 DnCNN、DRUNet、Restormer 等外部深度学习结果，不要在本项目中伪造输出。把外部模型对同一合成样本的输出放到：

```text
data/raw/learning_outputs/<method>/<sample_id>.png
```

然后重新运行：

```powershell
python scripts\11_run_exp06_learning_comparison.py --train-limit 4 --test-limit 4
```

## 每个样本应保存

```text
input.png
structure_gt.png              # 有真值时保存
metrics.csv
comparison_grid.png
<method>/
  output.png
  residual_vis.png
  intermediates/              # 有中间量时保存
```

SAC-SWF 必须保留：

```text
beta.png
structure_conf.png
coherence.png
texture_energy.png
oscillation.png
direction_index.png
q_full.png
q_side.png
```

这些图用于解释算法行为，不能删除。

## 最小检查

```powershell
python -m compileall -q src scripts tests
python scripts\run_all_smoke.py
```

`run_all_smoke.py` 会用很小样本覆盖实验 1-7，用于检查代码入口和结果落盘是否完整。

## 禁止事项

- 禁止改目录结构后继续运行。
- 禁止只保存 SAC-SWF，不保存 baseline。
- 禁止覆盖原始数据。
- 禁止把没有外部模型输出的实验 6 写成“已完成深度网络对比”；可以写成“完成 patch_ridge 数据驱动基线对比”。
- 禁止把 `official_drunet` 或 `official_restormer` 的缺权重失败状态写成已复现成功。
- 禁止把 `exp05_benchmark` 的 course benchmark suite 写成官方 EPS Benchmark。
- 禁止把 `exp07_paper_reproduction` 写成原作者代码 bit-exact 复现。
- 禁止只挑成功样例，报告中必须保留成功、一般和失败样例。
