"""
中心极限定理演示：从均匀分布采样，样本均值趋向正态分布

Central Limit Theorem: Sample means from uniform distribution converge to normal distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def demonstrate_clt(sample_sizes=[1, 2, 5, 10, 30, 100], num_experiments=10000):
    """
    演示中心极限定理

    Args:
        sample_sizes: 不同的样本大小 n
        num_experiments: 每个样本大小重复实验的次数
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    # 原始均匀分布参数 U(0, 1)
    a, b = 0, 1
    # 均匀分布的理论均值和标准差
    uniform_mean = (a + b) / 2  # 0.5
    uniform_std = (b - a) / np.sqrt(12)  # ≈ 0.289

    for idx, n in enumerate(sample_sizes):
        ax = axes[idx]

        # 进行 num_experiments 次实验，每次从均匀分布抽取 n 个样本
        sample_means = []
        for _ in range(num_experiments):
            samples = np.random.uniform(a, b, n)
            sample_means.append(np.mean(samples))

        sample_means = np.array(sample_means)

        # 绘制样本均值的直方图
        ax.hist(
            sample_means,
            bins=50,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="white",
            label="样本均值分布",
        )

        # 根据 CLT，样本均值的理论分布
        # 均值: μ = 0.5
        # 标准差: σ/√n
        theoretical_std = uniform_std / np.sqrt(n)
        x = np.linspace(sample_means.min(), sample_means.max(), 200)
        theoretical_pdf = stats.norm.pdf(x, uniform_mean, theoretical_std)

        ax.plot(
            x,
            theoretical_pdf,
            "r-",
            linewidth=2,
            label=f"理论正态分布 N({uniform_mean}, {theoretical_std:.3f}²)",
        )

        # 计算统计量
        actual_mean = np.mean(sample_means)
        actual_std = np.std(sample_means)

        ax.set_title(f"样本大小 n = {n}", fontsize=12, fontweight="bold")
        ax.set_xlabel("样本均值", fontsize=10)
        ax.set_ylabel("概率密度", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

        # 添加统计信息文本框
        stats_text = f"实际均值: {actual_mean:.4f}\n理论均值: {uniform_mean:.4f}\n实际标准差: {actual_std:.4f}\n理论标准差: {theoretical_std:.4f}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.suptitle(
        "中心极限定理演示：均匀分布 U(0,1) 的样本均值分布\n"
        f"(每个样本大小重复 {num_experiments:,} 次实验)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        "f:/Projects/ProbabilityTheory/clt_demonstration.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    print("=" * 60)
    print("中心极限定理 (Central Limit Theorem)")
    print("=" * 60)
    print("""
无论总体服从什么分布，只要满足一定条件，
样本均值的分布都会随着样本量的增加而趋向于正态分布。

对于均匀分布 U(0, 1)：
  - 总体均值 μ = 0.5
  - 总体标准差 σ = 1/√12 ≈ 0.289

根据 CLT，n 个样本的均值 X̄ 近似服从：
  X̄ ~ N(μ, σ²/n) = N(0.5, 0.0833/n)

从图中可以看到：
  1. n=1 时，分布就是原始的均匀分布
  2. 随着 n 增大，分布越来越接近正态分布（钟形曲线）
  3. n 增大时，标准差减小（分布更集中在均值附近）
  4. 通常 n≥30 时，正态近似已经相当好
""")


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 运行演示
    demonstrate_clt()
