import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


# =====================================================
# Hampel Filter
# =====================================================

def hampel_filter(
    signal,
    window_size=5,
    n_sigma=3
):
    """
    Hampel filter

    signal:
        (T,)

    return:
        filtered signal
        changed frame index
    """

    filtered = signal.copy()

    changed = []

    T = len(signal)


    for i in range(T):

        start = max(0, i-window_size)
        end = min(T, i+window_size+1)

        window = signal[start:end]


        median = np.median(window)

        mad = np.median(
            np.abs(window - median)
        )


        if mad < 1e-8:
            continue


        threshold = (
            n_sigma *
            1.4826 *
            mad
        )


        if abs(signal[i]-median) > threshold:

            filtered[i] = median

            changed.append(i)


    return filtered, changed



# =====================================================
# Apply Hampel to FLAME
# =====================================================

def apply_hampel(params):

    refined = params.copy()


    report = {}


    # ======================
    # head pose
    # 50:53
    # ======================

    head_changed=set()


    for dim in range(50,53):

        refined[:,dim], changed = hampel_filter(
            params[:,dim],
            window_size=7,
            n_sigma=4
        )

        head_changed.update(changed)



    report["head_pose"] = {

        "range":
            "50:53",

        "changed_count":
            len(head_changed),

        "changed_frames":
            sorted(list(head_changed)),

        "ratio":
            len(head_changed)/len(params)

    }



    # ======================
    # jaw pose
    # 53:56
    # ======================

    jaw_changed=set()


    for dim in range(53,56):

        refined[:,dim], changed = hampel_filter(
            params[:,dim],
            window_size=5,
            n_sigma=3
        )

        jaw_changed.update(changed)



    report["jaw_pose"] = {

        "range":
            "53:56",

        "changed_count":
            len(jaw_changed),

        "changed_frames":
            sorted(list(jaw_changed)),

        "ratio":
            len(jaw_changed)/len(params)

    }


    return refined, report



# =====================================================
# Plot
# =====================================================

def plot_compare(
    raw,
    refined,
    output_dir
):

    names = {

        50:"head_pose_x",
        51:"head_pose_y",
        52:"head_pose_z",

        53:"jaw_pose_x",
        54:"jaw_pose_y",
        55:"jaw_pose_z"
    }


    for dim,name in names.items():


        plt.figure(
            figsize=(12,4)
        )


        plt.plot(
            raw[:,dim],
            label="raw"
        )


        plt.plot(
            refined[:,dim],
            label="hampel"
        )


        plt.title(name)

        plt.legend()

        plt.grid()


        plt.savefig(
            output_dir /
            f"{name}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()



# =====================================================
# Main
# =====================================================

if __name__ == "__main__":


    # 修改这里
    input_file = Path(
        "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/test_data/npy/100248_0_4.npy"
    )


    output_dir = Path(
        "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/tests/flame_hampel/hampel_result"
    )


    output_dir.mkdir(
        exist_ok=True
    )


    params = np.load(
        input_file
    )


    print(
        "Input shape:",
        params.shape
    )


    assert params.shape[1] == 62



    refined, report = apply_hampel(
        params
    )


    np.save(
        output_dir /
        "hampel.npy",
        refined
    )


    with open(
        output_dir /
        "hampel_report.json",
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


    plot_compare(
        params,
        refined,
        output_dir
    )


    print("\n===== Hampel Result =====")

    print(json.dumps(
        report,
        indent=4
    ))

    print(
        "\nSaved:",
        output_dir
    )

    