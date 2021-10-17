'''
A module that can create animated plots using matplotlib.animation from the datastore of en expreiment
It also can save selected frames as png images
'''
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm
import os

from sklearn.preprocessing import minmax_scale

# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def scale_stack(stack):
    stack = reject_outliers(stack, 3.5)
    stack = minmax_scale(stack, feature_range=(0, 1))
    return stack

def animate(exp):
    fig, ax = plt.subplots()
    ims = []
    if exp._use_roi:
        stack = exp.norm_stack.swapaxes(0, 2)
    else:
        stack = exp.stack.swapaxes(0, 2)

    stack = scale_stack(stack)
    for index, frame in enumerate(stack):
        ax.set_title(f'Image shape: {frame.shape}')
        im = ax.imshow(frame, animated = True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True, repeat_delay = 3000)
    plt.show()

def masked_animation(stack, vmin = None, vmax = None):
    fig, ax = plt.subplots()
    ims = []
    for index, frame in enumerate(stack):
        im = ax.imshow(frame, animated = True, vmin = vmin, vmax = vmax, cmap = cm.get_cmap('viridis'))
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval = 100, blit = True, repeat_delay = 3000)
    plt.show()

def export_frames(exp, frame_list):
    if exp._use_roi:
        s = exp.norm_stack
    else:
        s = exp.stack
        print('Printing from stack')
    s = scale_stack(s)
    for index in frame_list:
        try:
            nparray = s[:, :, index]
            nparray -= nparray.min()
            nparray /= nparray.max()
            nparray *= 255
            im = Image.fromarray(np.uint8(nparray)).convert('RGB')
            path = f'../frames/{exp.name}-Frame-{index}.png'
            im.save(path)
        except Exception as e:
            print(e)
            continue

def export_frames_plot(exp, frame_list):
    if exp._use_roi:
        s = exp.norm_stack
    else:
        s = exp.stack
        print('Printing from stack')
    s = scale_stack(s)
    for index in frame_list:
        path = f'./{exp.name}-Frame-{index}.png'
        try:
            nparray = s[:, :, index]
            vmin = s.min()
            vmax = s.max()
            plt.imsave(path, nparray, vmin = vmin, vmax = vmax)
        except Exception as e:
            print(e)
            continue



def get_avg_stack_frames(exp):
    return exp.stack




