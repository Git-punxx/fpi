import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import h5py


#TODO 1. Identify island and find their contour/convex https://github.com/tirthajyoti/Scikit-image-processing/blob/master/Finding_contours.ipynb
#TODO Be able to select an island
#TODO 2. Create a mask using the selected island
#TODO 3. Overlay a tinted mask over animated frames from the average stack and update the timeline. Be able to export a selected region of frames

fname = 'F:\Data\datastore_20190924_1613_0.h5'

def load_data(fname):
    with h5py.File(fname) as f:
        avg_stack = f['df']['resp_map'][()]
        stack = f['df']['stack'][()]
        timeline = f['df']['avg_df'][()]
    return avg_stack, stack, timeline

def pil_to_cv(img):
    cv = np.array(img)
    return cv[:, :, ::-1].copy()


def normalize(arr):
    min_val = arr.min()
    arr = arr - min_val
    arr = np.uint8(arr*255/arr.max())
    return arr

def find_contours(arr):
    # Use binary images -> apply canny edge detection or threshold
    # Copy the source image because findContours modifies it
    # Object should be white and background should be black
    arr = normalize(arr)
    imgray = cv.cvtColor(arr, cv.COLOR_BGR2GRAY)
    ret, thres = cv.threshold(imgray, 127, 255, 0)
    image, contours = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.drawContours(imgray, contours, -1, (0, 255, 0), 3)
    plt.imshow(img)
    plt.show()

def image_threshold(df: np.ndarray, threshold: float) :
    mask = create_mask(df, threshold)
    resp = np.where(mask, df, 0)
    total_pixel = np.count_nonzero(mask)
    return resp, total_pixel


def create_mask(arr, threshold):
    return arr > arr.max() * (threshold/100.)

def masked_response(norm_stack, mask):
    x, y, z = norm_stack.shape
    mask = np.tile(mask, [z, 1, 1])
    df = np.where(mask.T, norm_stack, np.nan)
    df_avg = np.nanmean(df, axis = (0,1))
    return df_avg


if __name__ == '__main__':
    image, frame_stack, timeline = load_data(fname)
    img = Image.fromarray(image).convert('RGB')
    find_contours(img)
