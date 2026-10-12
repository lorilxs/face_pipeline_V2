import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def calc_rot_delta_deg(rotvec_seq):
    T = rotvec_seq.shape[0]
    rot_mats = Rotation.as_matrix(Rotation.from_rotvec(rotvec_seq))
    delta_deg_list = []
    for t in range(T - 1):
        Rt = rot_mats[t]
        Rt1 = rot_mats[t + 1]
        R_rel = Rt.T @ Rt1
        theta_rad = np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec())
        delta_deg_list.append(np.rad2deg(theta_rad))
    return np.array(delta_deg_list)


def compare_two_motion_npy(path_a: str, path_b: str, save_prefix="compare", vis_dir="results/vis/compare"):
    """
    对比两个 [T,62] motion npy 的 head / jaw 帧间旋转抖动
    dim50~52: head rotvec
    dim53~55: jaw rotvec
    """
    os.makedirs(vis_dir, exist_ok=True)

    # 加载数据
    motion_a = np.load(path_a)
    motion_b = np.load(path_b)

    # 建议时序长度对齐（取最短）
    min_len = min(motion_a.shape[0], motion_b.shape[0])
    motion_a = motion_a[:min_len]
    motion_b = motion_b[:min_len]
    frame_idx = np.arange(min_len)

    # 提取 rotvec
    head_a = motion_a[:, 50:53]           #head_a = motion_a[:, 50:53]
    jaw_a  = motion_a[:, 53:56]            #jaw_a  = motion_a[:, 53:56]
    head_b = motion_b[:, 100:103]         # head_b = motion_b[:, 50:53]
    jaw_b  = motion_b[:, 103:106]           #jaw_b  = motion_b[:, 53:56]

    # 计算帧间delta
    head_d_a = calc_rot_delta_deg(head_a)
    jaw_d_a  = calc_rot_delta_deg(jaw_a)
    head_d_b = calc_rot_delta_deg(head_b)
    jaw_d_b  = calc_rot_delta_deg(jaw_b)

    # ========== 打印指标对比 ==========
    print("=" * 60)
    print(f"File gt: {path_a}")
    print(f"File pred: {path_b}")
    print(f"有效帧数: {min_len}")
    print("=" * 60)
    print(f"【Head 抖动】")
    print(f"gt mean={np.mean(head_d_a):.4f}°, std={np.std(head_d_a):.4f}°")
    print(f"pred mean={np.mean(head_d_b):.4f}°, std={np.std(head_d_b):.4f}°")
    print("-" * 40)
    print(f"【Jaw 抖动】")
    print(f"gt mean={np.mean(jaw_d_a):.4f}°, std={np.std(jaw_d_a):.4f}°")
    print(f"pred mean={np.mean(jaw_d_b):.4f}°, std={np.std(jaw_d_b):.4f}°")
    print("=" * 60)

    # ========== 绘图对比：Head delta ==========
    plt.figure(figsize=(14, 4), dpi=120)
    plt.plot(head_d_a, label="gt Head Delta", c="#2980b9", linewidth=1.1)
    plt.plot(head_d_b, label="pred Head Delta", c="#e74c3c", linewidth=1.1)
    plt.title("Head Frame-to-Frame Rot Delta Compare (°)")
    plt.xlabel("Frame")
    plt.ylabel("Delta Rot (°)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{save_prefix}_head_delta.png"))

    # ========== 绘图对比：Jaw delta ==========
    plt.figure(figsize=(14, 4), dpi=120)
    plt.plot(jaw_d_a, label="gt Jaw Delta", c="#f39c12", linewidth=1.1)
    plt.plot(jaw_d_b, label="pred Jaw Delta", c="#9b59b6", linewidth=1.1)
    plt.title("Jaw Frame-to-Frame Rot Delta Compare (°)")
    plt.xlabel("Frame")
    plt.ylabel("Delta Rot (°)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{save_prefix}_jaw_delta.png"))

    # ========== 合并一张总图（推荐用于汇报） ==========
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,7), dpi=120)
    ax1.plot(head_d_a, label="gt Head", c="#2980b9")
    ax1.plot(head_d_b, label="pred Head", c="#e74c3c")
    ax1.set_title("Head Delta Compare")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(jaw_d_a, label="gt Jaw", c="#f39c12")
    ax2.plot(jaw_d_b, label="pred Jaw", c="#9b59b6")
    ax2.set_title("Jaw Delta Compare")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{save_prefix}_head_jaw_compare.png"))

    print(f"对比图保存至: {vis_dir}")
    return {
        "gt": {"head_mean": np.mean(head_d_a), "head_std": np.std(head_d_a),
               "jaw_mean": np.mean(jaw_d_a), "jaw_std": np.std(jaw_d_a)},
        "pred": {"head_mean": np.mean(head_d_b), "head_std": np.std(head_d_b),
                 "jaw_mean": np.mean(jaw_d_b), "jaw_std": np.std(jaw_d_b)}
    }


if __name__ == "__main__":
    # ===================== 修改这里两个npy路径 =====================
    path_gt = "/mnt/home/lxs/videos/HDTF/CelebV-flame-audio/flame/RD_Radio1_000_0_4.npy"
    path_pred = "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/outputs/video_RD_Radio1_000_8s/stage17_multimodal_export/flame/RD_Radio1_000_8s_scene_0000_clip_0000.npy"

    compare_two_motion_npy(
        path_a=path_gt,
        path_b=path_pred,
        save_prefix="gt_vs_pred",
        vis_dir="/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/tests/flame_hampel/EMOCA_hampel_EMICA_RD_Radio1_000_0_8s"
    )