"""
交互式中心极限定理演示
每次点击采样按钮，从均匀分布中采样 n 个值并计算均值
随着点击次数增加，逐渐呈现正态分布

Interactive Central Limit Theorem Demonstration
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy import stats

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


class InteractiveCLT:
    def __init__(self, sample_size=30, samples_per_click=5):
        """
        交互式 CLT 演示

        Args:
            sample_size: 每次采样的样本大小 n（从均匀分布抽取 n 个值计算均值）
            samples_per_click: 每次点击采样的次数
        """
        self.sample_size = sample_size
        self.samples_per_click = samples_per_click
        self.sample_means = []  # 存储所有样本均值

        # 均匀分布参数 U(0, 1)
        self.a, self.b = 0, 1
        self.uniform_mean = (self.a + self.b) / 2  # 0.5
        self.uniform_std = (self.b - self.a) / np.sqrt(12)  # ≈ 0.289

        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25)  # 为按钮和滑块留出空间

        # 创建按钮
        ax_sample = plt.axes([0.3, 0.05, 0.15, 0.06])
        self.btn_sample = Button(
            ax_sample, "采样", color="lightblue", hovercolor="steelblue"
        )
        self.btn_sample.on_clicked(self.on_sample_click)

        ax_sample10 = plt.axes([0.5, 0.05, 0.15, 0.06])
        self.btn_sample10 = Button(
            ax_sample10, "采样 x10", color="lightgreen", hovercolor="green"
        )
        self.btn_sample10.on_clicked(self.on_sample10_click)

        ax_reset = plt.axes([0.7, 0.05, 0.15, 0.06])
        self.btn_reset = Button(ax_reset, "重置", color="lightcoral", hovercolor="red")
        self.btn_reset.on_clicked(self.on_reset_click)

        # 创建滑块 - 调整样本大小 n
        ax_slider_n = plt.axes([0.25, 0.12, 0.5, 0.03])
        self.slider_n = Slider(
            ax_slider_n, "样本大小 n", 1, 100, valinit=sample_size, valstep=1
        )
        self.slider_n.on_changed(self.on_n_change)

        # 初始绘制
        self.update_plot()

    def take_samples(self, num_samples):
        """进行采样"""
        for _ in range(num_samples):
            # 从均匀分布抽取 n 个样本
            samples = np.random.uniform(self.a, self.b, self.sample_size)
            # 计算均值并存储
            self.sample_means.append(np.mean(samples))

    def on_sample_click(self, event):
        """点击采样按钮"""
        self.take_samples(self.samples_per_click)
        self.update_plot()

    def on_sample10_click(self, event):
        """点击采样 x10 按钮"""
        self.take_samples(self.samples_per_click * 10)
        self.update_plot()

    def on_reset_click(self, event):
        """点击重置按钮"""
        self.sample_means = []
        self.update_plot()

    def on_n_change(self, val):
        """滑块改变样本大小"""
        self.sample_size = int(val)
        # 重置数据，因为样本大小改变了
        self.sample_means = []
        self.update_plot()

    def update_plot(self):
        """更新图形"""
        self.ax.clear()

        num_means = len(self.sample_means)
        theoretical_std = self.uniform_std / np.sqrt(self.sample_size)

        if num_means == 0:
            # 没有数据时显示提示
            self.ax.text(
                0.5,
                0.5,
                '点击 "采样" 按钮开始！\n\n'
                f"每次点击将从 U(0,1) 中抽取 {self.sample_size} 个样本，\n"
                "计算它们的均值并添加到直方图中。\n\n"
                "随着采样次数增加，\n"
                "你会看到样本均值逐渐呈现正态分布！",
                ha="center",
                va="center",
                fontsize=14,
                transform=self.ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
            )
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
        else:
            # 绘制直方图
            # 动态调整 bins 数量
            bins = min(50, max(10, num_means // 5))

            self.ax.hist(
                self.sample_means,
                bins=bins,
                density=True,
                alpha=0.7,
                color="steelblue",
                edgecolor="white",
                label=f"样本均值分布 (共 {num_means} 次)",
            )

            # 绘制理论正态分布曲线
            x = np.linspace(0, 1, 200)
            theoretical_pdf = stats.norm.pdf(x, self.uniform_mean, theoretical_std)
            self.ax.plot(
                x,
                theoretical_pdf,
                "r-",
                linewidth=2.5,
                label=f"理论正态 N({self.uniform_mean}, {theoretical_std:.4f}^2)",
            )

            # 绘制实际均值线
            actual_mean = np.mean(self.sample_means)
            actual_std = np.std(self.sample_means) if num_means > 1 else 0
            self.ax.axvline(
                actual_mean,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"实际均值: {actual_mean:.4f}",
            )

            # 添加统计信息
            stats_text = (
                f"采样次数: {num_means}\n"
                f"样本大小 n: {self.sample_size}\n"
                f"─────────────\n"
                f"实际均值: {actual_mean:.4f}\n"
                f"理论均值: {self.uniform_mean:.4f}\n"
                f"─────────────\n"
                f"实际标准差: {actual_std:.4f}\n"
                f"理论标准差: {theoretical_std:.4f}"
            )
            self.ax.text(
                0.02,
                0.98,
                stats_text,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            self.ax.set_xlabel("样本均值", fontsize=12)
            self.ax.set_ylabel("概率密度", fontsize=12)
            self.ax.legend(loc="upper right", fontsize=10)

            # 设置合理的 x 轴范围
            self.ax.set_xlim(
                max(0, self.uniform_mean - 4 * theoretical_std),
                min(1, self.uniform_mean + 4 * theoretical_std),
            )

        # 设置标题
        self.ax.set_title(
            f"中心极限定理交互演示\n"
            f"从均匀分布 U(0,1) 中抽取 n={self.sample_size} 个样本，观察样本均值的分布",
            fontsize=14,
            fontweight="bold",
        )

        self.fig.canvas.draw_idle()

    def show(self):
        """显示图形"""
        plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("中心极限定理交互演示")
    print("=" * 60)
    print("""
使用说明：
  1. 点击 "采样" 按钮进行 5 次采样
  2. 点击 "采样 x10" 按钮进行 50 次采样
  3. 使用滑块调整每次采样的样本大小 n
  4. 点击 "重置" 清空所有数据重新开始
  
观察：
  - 随着采样次数增加，直方图逐渐逼近红色的理论正态曲线
  - 样本大小 n 越大，分布越集中（标准差越小）
  - 即使 n 很小，只要采样次数足够多，也能看到钟形分布
""")

    demo = InteractiveCLT(sample_size=30, samples_per_click=5)
    demo.show()


if __name__ == "__main__":
    main()
