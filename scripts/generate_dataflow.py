"""Generate data flow diagram (left-to-right, Chinese labels, B/W)."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(r"C:/Users/15001/WorkBuddy/2026-07-05-18-21-50/reports/paper_figures_new")
OUT.mkdir(parents=True, exist_ok=True)

# 支持中文
plt.rcParams.update({
    "font.family": "SimHei",
    "font.size": 10,
    "savefig.dpi": 220,
    "axes.unicode_minus": False,
})


def box(ax, x, y, w, h, text):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                    facecolor="white", edgecolor="black", lw=1.2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0))


def main():
    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    bw, bh = 1.8, 0.7  # box width, height
    # 三行从左到右
    # 行1 (y=3.8): 结构张量路径
    # 行2 (y=2.2): 纹理指示量路径
    # 行3 (y=0.6): 候选路径

    x0 = 0.3
    # 输入
    box(ax, x0, 2.2, 1.5, 0.7, "输入图像 I")
    # 从I分出三条箭头
    arrow(ax, x0+1.5, 2.55, x0+2.0, 4.15)  # 到行1
    arrow(ax, x0+1.5, 2.55, x0+2.0, 2.55)  # 到行2
    arrow(ax, x0+1.5, 2.55, x0+2.0, 0.95)  # 到行3

    # 行1: 亮度图 -> 结构张量 -> 特征值 -> R
    x = 2.1
    box(ax, x, 3.8, bw, bh, "亮度图 Y\n+ 梯度")
    arrow(ax, x+bw, 4.15, x+2.3, 4.15)
    x = 4.3
    box(ax, x, 3.8, bw, bh, "结构张量 Jρ")
    arrow(ax, x+bw, 4.15, x+2.3, 4.15)
    x = 6.5
    box(ax, x, 3.8, bw, bh, "特征值\n结构强度 E\n一致性 C")
    arrow(ax, x+bw, 4.15, x+2.3, 4.15)
    x = 8.7
    box(ax, x, 3.8, bw, bh, "结构置信度\nR = E*C")

    # 行2: 高频残差 -> 振荡O -> β
    x = 2.1
    box(ax, x, 2.2, bw, bh, "高频残差\nRh = Y - G*Y")
    arrow(ax, x+bw, 2.55, x+2.3, 2.55)
    x = 4.3
    box(ax, x, 2.2, bw, bh, "高频能量 H\n振荡 O = H*(1-C)")
    arrow(ax, x+bw, 2.55, x+2.3, 2.55)
    x = 6.5
    box(ax, x, 2.2, bw, bh, "组合权重\nb = clip(1-R+g*O)")
    # R 也连到 β
    arrow(ax, 9.6, 3.8, 7.4, 2.9)  # R -> β

    # 行3: full候选 -> 8方向side候选 -> 软聚合QS
    x = 2.1
    box(ax, x, 0.6, bw, bh, "full-window\n引导滤波候选 QF")
    arrow(ax, x+bw, 0.95, x+2.3, 0.95)
    x = 4.3
    box(ax, x, 0.6, bw, bh, "8方向side-window\n候选 Q(s)")
    arrow(ax, x+bw, 0.95, x+2.3, 0.95)
    x = 6.5
    box(ax, x, 0.6, bw, bh, "软选择聚合\nQS = Σα·Q(s)")

    # β 和 QF, QS -> 输出
    box(ax, 9.5, 1.2, 2.2, 0.9, "输出 S =\nb*QF+(1-b)*QS")
    arrow(ax, 7.4, 2.2, 9.5, 1.95)  # β -> 输出
    arrow(ax, 8.3, 0.95, 9.5, 1.45)  # QS -> 输出
    arrow(ax, 3.9, 0.95, 9.5, 1.45)  # QF -> 输出 (长箭头，弯一下)
    # QF 直接到输出，画一条折线
    arrow(ax, 3.0, 0.6, 3.0, 0.2)
    arrow(ax, 3.0, 0.2, 10.6, 0.2)
    arrow(ax, 10.6, 0.2, 10.6, 1.2)

    fig.savefig(OUT / "code_dataflow.png", bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)
    print("[saved] code_dataflow.png")


if __name__ == "__main__":
    main()
