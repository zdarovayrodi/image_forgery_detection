from typing import Dict

import cv2
import numpy as np


def compute_statistics(channel) -> Dict[str, float]:
    mean = np.mean(channel)
    median = np.median(channel)
    quantile_25 = np.percentile(channel, 25)
    quantile_75 = np.percentile(channel, 75)
    std = np.std(channel)

    return {
        "mean": mean,
        "median": median,
        "std": std,
        "quantile_25": quantile_25,
        "quantile_75": quantile_75,
    }


def compute_rgb_statistics(image) -> Dict[str, float]:
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    r_stats = compute_statistics(r_channel)
    g_stats = compute_statistics(g_channel)
    b_stats = compute_statistics(b_channel)

    return {
        **{f"r_{k}": v for k, v in r_stats.items()},
        **{f"g_{k}": v for k, v in g_stats.items()},
        **{f"b_{k}": v for k, v in b_stats.items()},
    }


def compute_hsl_statistics(image) -> Dict[str, float]:
    hsl_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    h_channel = hsl_image[:, :, 0]
    s_channel = hsl_image[:, :, 2]
    l_channel = hsl_image[:, :, 1]

    h_stats = compute_statistics(h_channel)
    s_stats = compute_statistics(s_channel)
    l_stats = compute_statistics(l_channel)

    return {
        **{f"hsl_h_{k}": v for k, v in h_stats.items()},
        **{f"hsl_s_{k}": v for k, v in s_stats.items()},
        **{f"hsl_l_{k}": v for k, v in l_stats.items()},
    }


def compute_hsv_statistics(image) -> Dict[str, float]:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]

    h_stats = compute_statistics(h_channel)
    s_stats = compute_statistics(s_channel)
    v_stats = compute_statistics(v_channel)

    return {
        **{f"hsv_h_{k}": v for k, v in h_stats.items()},
        **{f"hsv_s_{k}": v for k, v in s_stats.items()},
        **{f"hsv_v_{k}": v for k, v in v_stats.items()},
    }


def compute_ycbcr_statistics(image) -> Dict[str, float]:
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    y_channel = ycbcr_image[:, :, 0]
    cb_channel = ycbcr_image[:, :, 1]
    cr_channel = ycbcr_image[:, :, 2]

    y_stats = compute_statistics(y_channel)
    cb_stats = compute_statistics(cb_channel)
    cr_stats = compute_statistics(cr_channel)

    return {
        **{f"y_{k}": v for k, v in y_stats.items()},
        **{f"cb_{k}": v for k, v in cb_stats.items()},
        **{f"cr_{k}": v for k, v in cr_stats.items()},
    }
