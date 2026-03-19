import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
import matplotlib as mpl

# ================= 图形字体设置 =================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# ================= 散点颜色（按你最新要求）=================
emotion_colors = {
    -1: "#b25f79",  # Negative → 紫色
     0: "#5C6572",  # Neutral → 灰色
     1: "#669877"   # Positive → 绿色
}

# colorbar 渐变（负面 → 中性 → 正面）
emotion_cmap = LinearSegmentedColormap.from_list(
    "emotion_grad",
    ["#b25f79", "#5C6572", "#669877"]
)

# ====================================================================================
#                         绘制 epoch 的散点图
# ====================================================================================
def plot_one_epoch(epoch_id=30, save_only=False):

    features = np.load(f"features_epoch_{epoch_id}.npy")
    labels = np.load(f"labels_epoch_{epoch_id}.npy")

    mapped_labels = np.array([int(np.sign(v)) for v in labels])
    point_colors = np.array([emotion_colors[m] for m in mapped_labels])

    # t-SNE (2D)
    tsne = TSNE(n_components=2, random_state=42)
    X2d = tsne.fit_transform(features)

    # ====================== 绘制散点图 ==========================
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(
        X2d[:, 0], X2d[:, 1],
        c=point_colors,
        s=45,          # ★★ 再增大散点大小 ★★
        alpha=0.92,
        edgecolors='none'
    )

    # ====================== Colorbar ==========================
    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=emotion_cmap)
    cbar = plt.colorbar(mappable, ax=ax)

    cbar.set_label("Sentiment Polarity", fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    # ====================== 网格线 ==========================
    ax.grid(True, linestyle="--", alpha=1)

    # ====================== 标题 ======================
    ax.set_title("Model with DTM Replaced by Transformer", fontsize=30)

    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()

    # ====================== 保存图像 ======================
    fig.savefig("ReplaceDTM.png", dpi=600, bbox_inches='tight')
    print("[Saved] ReplaceDTM.png")

    if not save_only:
        plt.show()

    return fig
