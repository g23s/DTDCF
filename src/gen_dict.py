# gen_dict.py
import os
import numpy as np
from sklearn.cluster import KMeans

from data_loader import MSADataset
from config import get_config, get_args


def to_numpy_2d(feature):
    """
    将任意 list / numpy / tensor 转为 numpy，
    若是 1D 则直接返回，若 >=2D 则在 axis=0 做平均池化成 1D 向量。
    """
    try:
        arr = np.asarray(feature)
    except Exception:
        # 极端情况下再手动转换
        if hasattr(feature, 'cpu') and hasattr(feature, 'numpy'):
            arr = feature.cpu().numpy()
        else:
            arr = np.array(feature)

    if arr.ndim == 0:
        arr = arr.reshape(1, )
    if arr.ndim == 1:
        return arr.astype(np.float32)
    else:
        pooled = arr.mean(axis=0)
        return pooled.astype(np.float32)


def extract_modal_features(dataset, modal="vision"):
    """
    提取视觉模态特征
    使用平均池化得到固定长度向量
    MOSI/MOSEI 的 get_data 返回顺序是 (text, vision, audio, label...)
    """
    features = []
    for idx, sample in enumerate(dataset):  # 使用 enumerate 获取索引
        if modal == "vision":
            feat = sample[0][1]  # 视觉特征从 sample[0][1] 获取

            # 确保视觉特征是有效的，而不是情感标签
            print(f"[DEBUG] Vision feature at index {idx} type: {type(feat)}")
            if isinstance(feat, (int, float, np.int32, np.int64)):  # 如果是标量（情感标签）
                print(f"[WARNING] Vision feature at index {idx} is a scalar (probably label): {feat}")
                feat = np.zeros((35,))  # 假设视觉特征维度是 35，填充为零向量
            elif isinstance(feat, list):
                feat = np.array(feat)

            print(f"[DEBUG] Processed vision feature at index {idx} shape: {feat.shape}")

        else:
            raise ValueError("modal 必须是 'vision'")

        # 转换为 numpy 数组
        arr = np.array(feat)

        # Check for NaN values and handle them by filling with zeros
        if np.any(np.isnan(arr)):
            print(f"[WARNING] NaN values found in vision feature at index {idx}, filling with zeros.")
            arr = np.nan_to_num(arr)  # Fill NaN values with zero

        if arr.ndim == 1:  # 如果是 1D 向量
            pooled = arr
        else:
            pooled = arr.mean(axis=0)  # 平均池化
        features.append(pooled)

    return np.vstack(features)


def gen_npy(data, dataset="mosi", n_clusters=200, modal="vision", output_dir="./dict_npy"):
    """
    使用 KMeans 聚类并保存全局字典
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Ensure no NaN values exist in data before running KMeans
    if np.any(np.isnan(data)):
        print(f"[WARNING] NaN values found in the input data, filling with zeros.")
        data = np.nan_to_num(data)  # Fill NaN values with zero

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    centers = kmeans.cluster_centers_

    # Save the clustering centers as a .npy file
    save_path = f"{output_dir}/kmeans_{dataset}-{n_clusters}_{modal}.npy"
    np.save(save_path, centers)
    print(f"[OK] 已保存 {modal} 字典: {save_path}  形状: {centers.shape}")


if __name__ == "__main__":
    args = get_args()
    # 你的 get_config 签名是 (dataset, mode, batch_size)
    train_config = get_config(args.dataset, mode='train', batch_size=args.batch_size)
    train_dataset = MSADataset(train_config)

    # Step 1: 视觉特征 (平均池化)
    vision_features = extract_modal_features(train_dataset, modal="vision")
    gen_npy(vision_features, dataset=args.dataset, n_clusters=200, modal="vision", output_dir="./dict_npy")