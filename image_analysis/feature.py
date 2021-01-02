import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt


def process_image(exp_values: np.ndarray, percent: float) -> tuple:
    roi = np.where(exp_values > exp_values.max() * percent/100., exp_values, 0.)
    total_pixels = np.count_nonzero(roi)
    return roi, total_pixels

def create_mask(df, threshold):
    print(f'Creating mask with threshold {threshold}')
    mask = df > df.max() * threshold/100.
    total_pixels = np.count_nonzero(mask)
    return mask, total_pixels


def masked_response(norm_stack, mask):
    print(f'Creating masked response')
    x, y, z = norm_stack.shape
    mask = np.tile(mask, [z, 1, 1])
    print(mask.T.shape)
    df = np.where(mask.T, norm_stack, np.nan)
    print(f'Total non zero in df: {np.count_nonzero(df)}')
    df_avg = np.nanmean(df, axis = (0,1))
    return df_avg


if __name__ == '__main__':
    df = np.array([])
