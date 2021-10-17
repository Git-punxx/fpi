# ampplitude 5max frames max +-5






#halfwidth

#area average baseline
def timecourse_area(timecourse, mean_baseline):
    pos = np.maximum(timecourse - mean_baseline, 0)
    return np.trapz(pos)

#peak latency (30-75)

# max value of each pixel
def max_pixel_value(stack):
    return stack.max(2)

# threshold

# different areas that responding on different timecourse

# all pixels that responding significantly >25%

# max projection of all pixels

# areas with fluorescent intensity >80% Fmax



import numpy as np
import h5py
from matplotlib import pyplot as plt
import modified_intrinsic.imaging as imaging
import matplotlib.animation as animation

DS_PATH = "ds3.h5"
COLORMAP = "viridis"


def load_ds(path):
    ds = h5py.File(path, "r")
    return ds

def pixel_area(stack, threshold, max_df):
    response_stack = np.where(stack > threshold * max_df)
    return np.count_nonzero(response_stack, (0,1))

def normalize(array):
    min_el = array.min()
    if min_el < 0:
        array += abs(min_el)
    return array/np.linalg.norm(array)

class ExpAnalysis:
    def __init__(self, datastore, percent = 95):
        print("Loading experiment")
        self.datastore = h5py.File(datastore, "r")
        self.stack = self.datastore.get("df").get("stack")[()]
        self.max_resp = self.datastore.get("df").get("max_project")
        self.resp, self.df = imaging.resp_map(self.stack, percent)
        self.normalized = self.normalize()
        print("Stack loaded")
        self.vmin = np.percentile(self.normalized, 5)
        self.vmax = np.percentile(self.normalized, 95)


    def amplitude(self):
        pass

    def halfwidth(self):
        pass

    def timecourse(self):
        # average df per frame
        return self.df.mean(0)

    def max_df(self):
        return self.df.max(1).mean()

    def timecourse_area(self):
        mean_baseline = np.full(self.timecourse().shape, self.mean_baseline())
        line = np.maximum(self.timecourse()[30:80] - mean_baseline[30:80], 0)
        return np.trapz(line)

    def stimulation_timecourse(self):
        return self.timecourse()[30:]

    def fluorescent_area(self, threshold = 0.85):
        threshold = self.max_df() * threshold
        response = np.where(self.stack > threshold, self.stack, 0)
        return np.count_nonzero(response, (0,1))

    def global_max(self):
        return self.stack.max()

    def global_min(self):
        return self.stack.min()

    def max_df_per_frame(self):
        return self.df.max(0)

    def min_df_per_frame(self):
        return self.df.min(0)

    def avg_df_per_frame(self):
        return self.df.mean(0)

    def peak_val(self):
        return self.stimulation_timecourse().max()

    def peak_latency(self):
        index = np.argmax(self.stimulation_timecourse())
        return index

    def amp_percent(self):
        return (self.max_resp - self.mean_baseline())/self.mean_baseline()

    def max_pixel_response(self):
        return self.stack.max(2)

    def mean_baseline(self):
        return self.timecourse()[0:30].mean()

    def get_frame(self, index):
        return self.stack[:, :, index]

    def get_normalized_frame(self, frame):
        frame = self.normalized[:, :, frame]
        return frame

    def summarize(self):
        for index in range(0, 80):

            frame = self.get_frame(index)
            masked = np.where(self.resp > 0, frame, np.NaN)
            print(np.nanmean(masked))

    def normalize(self):
        self.normalized = np.empty_like(self.stack)
        for i in range(80):
            norm  = normalize(self.stack[:,:,i])
            self.normalized[:, :, i] = norm
        return self.normalized

    def animate(self):
        fig, ax = plt.subplots()
        ims = []

        for index in range(80):
            im = ax.imshow(self.get_normalized_frame(index), vmin= self.vmin, vmax=self.vmax, cmap=COLORMAP, animated = True)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True, repeat_delay = 3000)

        plt.show()

    def plot_frame(self, index):
        frame = self.get_normalized_frame(index)
        plt.imshow(frame, cmap=COLORMAP, vmin=self.vmin, vmax=self.vmax)
        plt.show()


    def masked_animate(self):
        fig, ax = plt.subplots()
        ims = []

        for index in range(80):
            frame = self.get_normalized_frame(index)
            frame = np.where(self.resp > 0, frame, self.vmin)
            im = ax.imshow(frame, vmin=self.vmin, vmax=self.vmax, cmap=COLORMAP, animated=True)

            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=3000)
        plt.show()

    def plot_im_resp(self):
        plt.imshow(self.resp)
        plt.show()


    def peak_latency(self, start = 30, end = 75):
        # it must return a frame
        response_region = self.timecourse()[start:end]
        peak = np.argmax(response_region)
        peak_value = np.max(response_region)
        return peak + 30


    def halfwidth(self, no_baseline=30):
        response_curve = self.timecourse()
        baseline_val = self.mean_baseline()
        peak_val = self.peak_val()

        half_val = (peak_val - baseline_val) / 2

        med_line = np.zeros_like(response_curve[no_baseline:80])
        med_line[()] = half_val

        idx = np.argwhere(np.diff(np.sign(response_curve[no_baseline:80] - med_line))).flatten()
        if len(idx) < 2:
            return (0, 0), 0
        halfwidth_start, *_, halfwidth_end = idx
        print(halfwidth_start)
        print(halfwidth_end)
        print(f'Response value at {halfwidth_start} to {halfwidth_end} = {response_curve[idx + no_baseline]}')
        plt.plot(response_curve[no_baseline:])
        plt.grid()
        plt.plot(med_line)
        plt.show()

        return halfwidth_end - halfwidth_start

if __name__ == '__main__':
    exp = ExpAnalysis(DS_PATH, 90)
    exp.masked_animate()