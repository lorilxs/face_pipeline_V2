import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter



# =====================================================
# FLAME 62D definition
#
# 0:50   expression
# 50:53  head_pose
# 53:56  jaw_pose
# 56:62  gaze
#
# Only smooth pose
# =====================================================



# =====================================================
# Hampel Filter
# =====================================================

def hampel_filter(
    signal,
    window_size=5,
    n_sigma=3
):

    filtered = signal.copy()

    changed = []


    for i in range(len(signal)):

        start = max(
            0,
            i-window_size
        )

        end = min(
            len(signal),
            i+window_size+1
        )


        window = signal[start:end]


        median = np.median(window)


        mad = np.median(
            np.abs(window-median)
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
# Apply Hampel
# =====================================================

def apply_hampel(params):

    output = params.copy()


    report={}


    # -------------------
    # head pose
    # 50:53
    # -------------------

    head_changed=set()


    for dim in range(50,53):

        output[:,dim], changed = hampel_filter(
            params[:,dim],
            window_size=7,
            n_sigma=4
        )

        head_changed.update(changed)



    # -------------------
    # jaw pose
    # 53:56
    # -------------------

    jaw_changed=set()


    for dim in range(53,56):

        output[:,dim], changed = hampel_filter(
            params[:,dim],
            window_size=5,
            n_sigma=3
        )

        jaw_changed.update(changed)



    report={

        "head_pose_changed":
            len(head_changed),

        "jaw_pose_changed":
            len(jaw_changed)

    }


    return output, report



# =====================================================
# SG
# =====================================================

def apply_sg(params):

    output=params.copy()


    for dim in range(50,53):

        output[:,dim]=savgol_filter(
            params[:,dim],
            window_length=7,
            polyorder=2
        )


    for dim in range(53,56):

        output[:,dim]=savgol_filter(
            params[:,dim],
            window_length=5,
            polyorder=2
        )


    return output



# =====================================================
# Plot
# =====================================================

def plot_compare(
    raw,
    hampel,
    sg,
    hampel_sg,
    output_dir
):


    names={

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
            hampel[:,dim],
            label="hampel"
        )


        plt.plot(
            sg[:,dim],
            label="sg"
        )


        plt.plot(
            hampel_sg[:,dim],
            label="hampel+sg"
        )


        plt.title(name)

        plt.legend()

        plt.grid()


        plt.savefig(
            output_dir /
            f"{name}_4_compare.png",
            dpi=150,
            bbox_inches="tight"
        )


        plt.close()



# =====================================================
# Main
# =====================================================

if __name__=="__main__":


    # =====================
    # 修改这里
    # =====================

    input_file = Path(
        "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/test_data/npy/100248_0_4.npy"
    )


    output_dir = Path(
        "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/tests/flame_hampel/hampel_sg_jaw_window5_compare"
    )


    # =====================


    output_dir.mkdir(
        exist_ok=True
    )


    raw=np.load(
        input_file
    )


    print(
        "Input:",
        raw.shape
    )


    assert raw.shape[1]==62



    # -------------------
    # 1 Hampel
    # -------------------

    hampel, hampel_report = apply_hampel(
        raw
    )



    # -------------------
    # 2 SG only
    # -------------------

    sg = apply_sg(
        raw
    )



    # -------------------
    # 3 Hampel + SG
    # -------------------

    hampel_sg = apply_sg(
        hampel
    )



    # save

    np.save(
        output_dir/"raw.npy",
        raw
    )


    np.save(
        output_dir/"hampel.npy",
        hampel
    )


    np.save(
        output_dir/"sg.npy",
        sg
    )


    np.save(
        output_dir/"hampel_sg.npy",
        hampel_sg
    )



    report={

        "hampel":
            hampel_report,


        "sg":{

            "window_length":5,

            "polyorder":2

        },


        "processed":
        "head_pose(50:53)+jaw_pose(53:56)"

    }



    with open(
        output_dir/"report.json",
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )



    plot_compare(
        raw,
        hampel,
        sg,
        hampel_sg,
        output_dir
    )


    print("\n===== DONE =====")

    print(
        json.dumps(
            report,
            indent=4
        )
    )


    print(
        "Saved:",
        output_dir
    )