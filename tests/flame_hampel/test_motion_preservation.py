import numpy as np
from pathlib import Path
import json



# =====================================================
# FLAME 62D
#
# 0:50 expression
# 50:53 head_pose
# 53:56 jaw_pose
# 56:62 gaze
#
# Compare raw vs refined
# =====================================================



def calc_stats(raw, refined, name):


    result={}


    raw_data = raw
    ref_data = refined



    # -----------------------------
    # range
    # -----------------------------

    raw_min = np.min(raw_data, axis=0)
    raw_max = np.max(raw_data, axis=0)

    ref_min = np.min(ref_data, axis=0)
    ref_max = np.max(ref_data, axis=0)


    raw_range = raw_max - raw_min
    ref_range = ref_max - ref_min



    range_ratio = (
        ref_range /
        (raw_range + 1e-8)
    )


    # -----------------------------
    # max absolute value
    # -----------------------------

    raw_abs_max = np.max(
        np.abs(raw_data),
        axis=0
    )

    ref_abs_max = np.max(
        np.abs(ref_data),
        axis=0
    )


    max_ratio = (
        ref_abs_max /
        (raw_abs_max + 1e-8)
    )



    # -----------------------------
    # motion energy
    # sum delta
    # -----------------------------

    raw_delta = np.diff(
        raw_data,
        axis=0
    )

    ref_delta = np.diff(
        ref_data,
        axis=0
    )


    raw_energy = np.sum(
        np.abs(raw_delta)
    )


    ref_energy = np.sum(
        np.abs(ref_delta)
    )


    energy_ratio = (
        ref_energy /
        (raw_energy + 1e-8)
    )



    result[name]={

        "raw_range":
            raw_range.tolist(),

        "ref_range":
            ref_range.tolist(),

        "range_ratio":
            range_ratio.tolist(),


        "raw_abs_max":
            raw_abs_max.tolist(),

        "ref_abs_max":
            ref_abs_max.tolist(),

        "max_ratio":
            max_ratio.tolist(),


        "raw_motion_energy":
            float(raw_energy),

        "ref_motion_energy":
            float(ref_energy),

        "energy_ratio":
            float(energy_ratio)

    }


    return result




# =====================================================
# Main
# =====================================================


if __name__=="__main__":


    # 修改这里

    raw_file = Path(
        "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/test_data/npy/100248_0_4.npy"
    )


    refined_file = Path(
        "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/tests/flame_hampel/hampel_sg_jaw_window5_compare/hampel_sg.npy"
    )


    output_file = Path(
        "/mnt/home/lxs/face_pipeline/Face_Pipeline_V2/tests/flame_hampel/motion_preservation_result/motion_preservation_jaw_window5_report.json"
    )



    raw = np.load(
        raw_file
    )


    refined = np.load(
        refined_file
    )



    assert raw.shape == refined.shape


    print(
        "shape:",
        raw.shape
    )



    report={}



    # ---------------------
    # head
    # ---------------------

    report.update(
        calc_stats(
            raw[:,50:53],
            refined[:,50:53],
            "head_pose"
        )
    )


    # ---------------------
    # jaw
    # ---------------------

    report.update(
        calc_stats(
            raw[:,53:56],
            refined[:,53:56],
            "jaw_pose"
        )
    )



    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            report,
            f,
            indent=4
        )


    print(
        json.dumps(
            report,
            indent=4
        )
    )


    print(
        "\nSaved:",
        output_file
    )