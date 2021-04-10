import numpy as np
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
import h5py
from gui.animator import *
from image_analysis import util
from itertools import compress


#TODO 1. Identify island and find their contour/convex https://github.com/tirthajyoti/Scikit-image-processing/blob/master/Finding_contours.ipynb
#TODO Be able to select an island
#TODO 2. Create a mask using the selected island
#TODO 3. Overlay a tinted mask over animated frames from the average stack and update the timeline. Be able to export a selected region of frames

fname = 'F:\Data\datastore_20190924_1613_0.h5'

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def load_data(fname, roi):
    root = 'df'
    if roi:
        root = 'roi'
    with h5py.File(fname) as f:
        avg_stack = f[root]['resp_map'][()]
        stack = f[root]['stack'][()]
        timeline = f[root]['avg_df'][()]
    return avg_stack, stack, timeline

def pil_to_cv(img):
    cv = np.array(img)
    return cv[:, :, ::-1].copy()


def normalize(arr):
    '''
    Convert the values of an ndarray to 0-255 to use the array with cv to find the contours
    :param arr:ndarray
    :return: ndarray
    '''
    min_val = arr.min()
    arr = arr - min_val
    arr = np.uint8(arr*255/arr.max())
    return arr

def image_threshold(df: np.ndarray, threshold: float) :
    # probably not used anywhere :p
    mask = create_mask(df, threshold)
    resp = np.where(mask, df, 0)
    total_pixel = np.count_nonzero(mask)
    return resp, total_pixel


def create_mask(arr, threshold):
    # probably not used anywhere :p
    return arr > arr.max() * (threshold/100.)

active_contours = []

def find_contours(arr: np.ndarray, threshold: int, rect_length):
    '''
    This function is used in the gui to update the image when changing the threshold and when we build the masked stack
    where we use the mask it returns

    Parameters
    ----------
    arr: 2d ndarray. Here we use the resp_map that contains the average values of the experiment
    threshold: int. Percentage used in findContours
    Returns
    -------
    final_image: ndarray. An array with the contours drawn onto it
    mean_area: The mean of the contour area values
    max_area: The value of the max area
    min_area: The value of the min area
    area_count: How many areas the findContour returned
    mask: ndarray. A mask created using the contours found
    '''
    # Use binary images -> apply canny edge detection or threshold
    # Copy the source image because findContours modifies it
    # Object should be white and background should be black
    arr = normalize(arr.copy())
    threshold = int((threshold * 255) / 100)
    img = cv.cvtColor(arr, cv.COLOR_GRAY2BGR)
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thres= cv.threshold(imgray, threshold, 255, 0)
    contours = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]

    areas = [(index, cv.contourArea(cnt)) for index, cnt in enumerate(contours)]
    mean_area = np.mean(areas)
    max_area = np.max(areas)
    min_area = np.min(areas)
    area_count = len(areas)

    global active_contours
    active_contours = [contour for contour in contours if cv.contourArea(contour) > 200]
    active_contours.sort(key = lambda c: -cv.contourArea(c))


    for (index, c) in enumerate(active_contours):
        # compute the center of each contour
        M = cv.moments(c)
        if M['m00'] == 0:
            continue
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        # draw the a circle at the center of the contour
        text_offset = 5

        xs, xe, ys, ye = (cX - rect_length // 2, cX + rect_length // 2, cY - rect_length //2 , cY + rect_length //2)

        cv.rectangle(img, (xs, ys), (xe, ye), (255, 0, 0), 1)
        cv.putText(img, str(index), (xs - text_offset, ys - text_offset), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    mask = cv.drawContours(np.zeros_like(arr),contours, -1, (255), -1)
    final_image = cv.drawContours(img, contours, -1, GREEN, 1)
    return final_image, mean_area, max_area, min_area, area_count, mask

def check_hitbox(x, y):
    global active_contours
    hit = [util.in_hull((x, y), contour[:, 0, :]) for contour in active_contours]
    #util.plot_triangulation(contour[:, 0, :])
    box = list(compress(active_contours, hit))
    if not box:
        return None
    else:
        return box[0]

def masked_timeseries(norm_stack: np.ndarray, mask: np.ndarray):
    '''
    It is used to build the plot of the response for the selected pixels
    Parameters
    ----------
    norm_stack: ndaarray. It is loaded from the datastore
    mask: A boolean mask created by find_contours that selects only the pixels over a certain threshold
    Returns:
    -------
    avg_df: ndarary. The timeseries
    vmin: The minimum value of the active pixels on all frames
    vmax: The maximum value of the active pixels on all frames

    '''
    x, y, z = norm_stack.shape
    mask = np.stack((mask,) * z, axis = 2)
    df = np.where(mask != 0, norm_stack, np.full([x, y, z], np.nan))

    # Build the timeseries
    df_avg = np.nanmean(df, axis = (0,1))

    # Find min and max values for the masked areas
    vmin = df.min() # use them on colormap to mask the rest
    vmax = df.max()
    return df_avg, vmin, vmax

def build_masked_frames(threshold: int, roi = False):
    '''
    This function loads the norm_stack, the resp_map and the response from the datastore
    It then computes the masked timeseries, the min and max value of the resulting stack
    Parameters
    ----------
    threshold

    Returns
    -------

    '''
    avg_stack, stack, timeline = load_data(fname, roi)
    *rest, mask = find_contours(avg_stack, threshold)

    # The avg contains nans
    avg, vmin, vmax = masked_timeseries(stack, mask)
    x, y, z = stack.shape
    mask = np.tile(mask, [z, 1, 1])
    frames = np.where(mask, stack.T, 0)
    images = [util.PIL_image_from_array(frame) for index, frame in enumerate(frames)]
    return images, vmin, vmax, frames

def compute_rect_timeseries(response, rect_size = 15):
    '''
    It returns the slice of the stack that corresponds to the rectangle positioned on the centroid of the contour
    :param response: The stack of the experiments (ndarray 683*683*80)
    :return:
    '''
    # We select the contour with the bigger area
    live_cnt = active_contours[0]
    M = cv.moments(live_cnt)
    # See https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html
    if M['m00'] == 0:
        raise ValueError('m00 is 0 on compute_rect_timeseries')
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])
    # draw a rect at the center of the contour
    from_x, to_x, from_y, to_y = (cX, cX + rect_size, cY, cY + rect_size)

    return response[from_x: to_x, from_y: to_y, :]



def test_frame_ani():
    avg_stack, stack, timeline = load_data(fname)
    stack = normalize(stack)
    vmin = 0
    vmax = 255
    for index, frame in enumerate(stack.T):
        plt.imshow(frame)
        plt.show()


def play_animation(roi = False):
    stack, vmin, vmax, frames = build_masked_frames(20, roi)
    for frame in stack:
        cv.imshow('Some title', pil_to_cv(frame))
        if cv.waitKey(33) == 27:
            break #Esc to quit
    cv.destroyAllWindows()

if __name__ =='__main__':
    play_animation()

