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

def animate(exp):
    fig, ax = plt.subplots()
    ims = []
    if exp._use_roi:
        stack = exp.norm_stack.swapaxes(0, 2)
    else:
        stack = exp.stack.swapaxes(0, 2)
    for index, frame in enumerate(stack):
        ax.set_title(f'Image shape: {frame.shape}')
        im = ax.imshow(frame, animated = True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True, repeat_delay = 3000)
    plt.show()

def export_frames(exp, frame_list):
    if exp._use_roi:
        s = exp.norm_stack
    else:
        s = exp.stack
        print('Printing from stack')
    for index in frame_list:
        try:
            nparray = s[:, :, index]
            nparray -= nparray.min()
            nparray /= nparray.max()
            nparray = cm.viridis(nparray)
            nparray *= 255
            im = Image.fromarray(np.uint8(nparray)).convert('RGB')
            path = f'../frames/{exp.name}-Frame-{index}.png'
            im.save(path)
        except Exception as e:
            print(e)
            continue


def get_avg_stack_frames(exp):
    return exp.stack




