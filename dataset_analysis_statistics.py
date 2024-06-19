import os

import cv2
import pandas as pd

from statistic_functions import (
    compute_hsl_statistics,
    compute_hsv_statistics,
    compute_rgb_statistics,
    compute_ycbcr_statistics,
)


def process_images(folder_path, is_real):
    face_stats = []

    debug_index = 0

    for root, _, files in os.walk(folder_path):
        for filename in files:
            try:
                face_img = cv2.imread(os.path.join(root, filename))
                rgb_stats = compute_rgb_statistics(face_img)
                hsl_stats = compute_hsl_statistics(face_img)
                hsv_stats = compute_hsv_statistics(face_img)
                ycbcr_stats = compute_ycbcr_statistics(face_img)

                combined_stats = {
                    "filename": filename,
                    **rgb_stats,
                    **hsl_stats,
                    **hsv_stats,
                    **ycbcr_stats,
                }

                face_stats.append(combined_stats)
                debug_index += 1
                if debug_index % 1000 == 0:
                    print(f"Doing {debug_index} iteration")
            except Exception as error:
                print(f"Exception: {error}")
                print(cv2.imread(os.path.join(root, filename)))

    df = pd.DataFrame(face_stats)
    df["real"] = 1 if is_real else 0
    df.to_csv(f"{folder_path}_stats.csv", index=False)
    print(f"Output to: {folder_path}_stats.csv")


process_images("./dataset/train/real", is_real=True)
process_images("./dataset/train/fake", is_real=False)
