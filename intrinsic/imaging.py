import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.patches import Rectangle
from matplotlib.cm import viridis, YlOrRd
from matplotlib.gridspec import GridSpec
import seaborn as sns
from intrinsic.stack import Stack
import numpy as np
import os
from tqdm import tqdm
from skimage.measure import block_reduce
from skimage.io import imread, imsave
from skimage.filters import median as med_filt
from skimage.filters import gaussian as gauss_filt
from skimage.morphology import disk
from skimage.color import grey2rgb
from skimage.measure import label, regionprops
from sklearn import mixture
from scipy.stats import pearsonr
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq
from scipy.stats import mannwhitneyu
import re
import h5py
from pathlib import Path
from typing import Union
from warnings import warn
import pandas as pd
from itertools import combinations, product
try:
    import moviepy.editor as mpy
    MOVIE_EXPORT = True
except ImportError:
    print("No movie exporting available")
    MOVIE_EXPORT = False

TIME_LABEL = 'Time (s)'
DEJAVU = 'DejaVu Serif'
ALL_PNG = '*.png'


def stim(t, t_on=30, tau_on=2, tau_off=4):
    # Time constants in frames
    return np.exp(-(t-t_on) / tau_off) - np.exp(-(t-t_on) / tau_on)


def img_to_uint8(img):
    img = np.float32(img.copy())
    return np.uint8(255 * (img - img.min()) / (img.max() - img.min()))


class ReducedStack(Stack):
    def __init__(self, path, pattern, binning=1):
        super().__init__(path, pattern)
        self.binning = binning
        self._previous_avg = None

    def get_pic(self, im):
        pic = super().get_pic(im)
        pic = block_reduce(pic, (self.binning, self.binning), np.mean)
        c_avg = pic.mean()
        if self._previous_avg is not None:
            ratio = c_avg / self._previous_avg
            if ratio > 1.5 or ratio < .7:
                pic = pic / ratio
                self._previous_avg = c_avg / ratio
        self._previous_avg = pic.mean()
        return pic


class TiffStack(Stack):
    def __init__(self, path):
        """
        Initialize an image stack

        Parameters
        ----------
        path: str
            Path to the serie of pictures
        """
        self._path = path
        # Open the Tiff Stack
        self._pics = imread(path)
        # List of the paths to all the stack images
        self.images = np.array([os.path.basename(path)+f'_{ix}' for ix in range(self._pics.shape[0])])

    def next(self):
        for im in self._pics:
            yield im

    def __getitem__(self, item):
        return self._pics[item, ...]


class Intrinsic(object):
    def __init__(self, path: Union[str, Path], pattern=ALL_PNG,
                 n_baseline=30, n_stim=30, n_recover=20, binning=1, exp_time=.1,
                 start=0, end=-1):
        self.path = Path(path)
        self.save_path = self.path / f'datastore_{self.path.name}.h5'
        if start != 0 or end != -1:
            self.save_path = self.path / f'datastore_{self.path.name}_{start}-{end}.h5'
        self.pattern = pattern
        self.n_baseline = n_baseline
        self.n_stim = n_stim
        self.n_recover = n_recover
        self.binning = binning
        self.exp_time = exp_time
        if pattern == ALL_PNG:
            self.trial_folders = self.get_trial_folders(self.path)
            if end != -1 or start != 0:
                end = len(self.trial_folders) if end == -1 else end
                self.trial_folders = self.trial_folders[start:end]
            stacks = [ReducedStack(trial, pattern, binning) for trial in self.trial_folders]
            self.stacks = [s for s in stacks if len(s) > self.n_baseline]
        else:
            tiff_files = self.get_tiff_list()
            if end != -1:
                tiff_files = tiff_files[start:end]
            self.stacks = [TiffStack(f) for f in tqdm(tiff_files, desc='Loading TIFF')]
            self.trial_folders = [self.path]

        self.l_base = np.array([])
        self.baseline = np.array([])
        self.avg_stack = np.array([])
        self._max_project = None
        self._resp = None
        self._norm_stack = None
        self.compute_baselines()

    def get_tiff_list(self):
        regex = re.compile('([0-9]*)')
        l_path = np.array(list(self.path.glob(self.pattern)))
        l_filenames = [p.name for p in l_path
                       if p.name.split('.')[-1] == 'tif']

        try:
            index = [int(''.join(regex.findall(flnm)))
                     for flnm in l_filenames]
        except ValueError:
            print("No number detected in one file")
            return l_path
        # get the indexes of the last number in the file name
        index = np.argsort(index)
        return l_path[index]

    @staticmethod
    def get_trial_folders(path: Union[str, Path]):
        path = Path(path)
        regex = re.compile('([0-9]*)')

        trial_folders = [path / o
                         for o in os.listdir(path)
                         if (path / o).is_dir()]
        trial_folders = np.array(trial_folders)
        try:
            index = [int(''.join(regex.findall(folder.as_posix())))
                     for folder in trial_folders]
        except ValueError:
            print("No number detected in one folder")
            index = np.arange(len(trial_folders))
        index = np.argsort(index)
        trial_folders = trial_folders[index]

        return trial_folders

    def save_analysis(self):
        with h5py.File(self.save_path, 'w') as f:
            f.create_dataset('n_baseline', data=self.n_baseline)
            f.create_dataset('n_stim', data=self.n_stim)
            f.create_dataset('n_recover', data=self.n_recover)
            f.create_dataset('dt', data=self.exp_time)
            f.create_dataset('anat', data=self.stacks[0][0])
            trials_grp = f.create_group('trials')
            for s in self.stacks:
                trials_grp.create_dataset(os.path.basename(s.path), data=s.images.astype('S'))
            analysed_grp = f.create_group('df')
            analysed_grp.create_dataset('max_project', data=self.max_project())
            analysed_grp.create_dataset('stack', data=self.avg_stack)
            analysed_grp.create_dataset('response', data=self.find_resp())
            # _, centers, cov = self.id_sources()
            # analysed_grp.create_dataset('centers', data=centers)
            # analysed_grp.create_dataset('cov', data=cov)
         # imsave((self.path / 'overlay.png').as_posix(), self.overlay())

    def compute_baselines(self):
        self.l_base = [s[:self.n_baseline].mean(0)
                       for s in tqdm(self.stacks, desc='Computing baseline')]
        self.baseline = np.mean(self.l_base, 0)

    def average_trials(self, start=0, end=-1):
        max_frames = max([len(s) for s in self.stacks])
        frame_shape = self.stacks[0][0].shape
        self.avg_stack = np.zeros((frame_shape[0], frame_shape[1], max_frames))
        for ix_frame in tqdm(range(max_frames), desc='Average trial'):
            all_c_frame = [s[ix_frame]
                           for i_s, s in enumerate(self.stacks[start:end])
                           if (ix_frame < len(s))]
            all_c_frame = [x for x in all_c_frame if x.shape is not ()]
            c_frame = np.array([x for x in all_c_frame])
            avg_frame = c_frame.mean(0)
            avg_frame -= self.baseline
            avg_frame /= self.baseline
            avg_frame[np.isnan(avg_frame)] = 0
            self.avg_stack[:, :, ix_frame] = avg_frame

    def norm_stack(self):
        if len(self.avg_stack) == 0:
            self.average_trials()
        if self._norm_stack is None:
            self._norm_stack = normalize_stack(self.avg_stack, self.n_baseline)
        return self._norm_stack

    def max_project(self):
        if self._max_project is None:
            stack = self.norm_stack()
            stim_frames = stack[..., self.n_baseline:self.n_baseline+self.n_stim]
            self._max_project = np.nanmax(stim_frames, 2)
        return self._max_project

    def find_resp(self):
        if self._resp is not None:
            return self._resp
        self._resp = find_resp(self.norm_stack(), self.n_baseline)
        return self._resp

    def id_sources(self):
        return id_sources(self.n_baseline, self.norm_stack())

    def overlay(self, cm=YlOrRd):
        resp = self.find_resp()
        f_resp = clean_response(resp)
        overlaid = overlay(self.stacks[0][0], resp=f_resp)

        return overlaid

    def __repr__(self):
        return f'Imaging session with {len(self.trial_folders)} trials from {self.path}'


class Session(object):
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.file = h5py.File(self.path, 'r+')
        self._norm_stack = None
        self._duration = 10
        self._movie_stack = None

    def close(self):
        try:
            self.file.close()
        except ValueError:
            # File already closed
            pass

    @property
    def anat(self):
        try:
            anat_pic = self.file['anat'][()]
        except (KeyError, ValueError):
            anat_pic = np.zeros(self.avg_stack.shape[:2])
        return anat_pic

    @property
    def resp(self):
        return gauss_filt(self.file['df']['max_project'][()], 3)

    @property
    def centers(self):
        try:
            return self.file['df']['centers'][()]
        except KeyError:
            return None

    @property
    def comment(self):
        if 'comment' in self.file.keys():
            return self.file['comment'][()]
        else:
            return ''

    @comment.setter
    def comment(self, value):
        if 'comment' in self.file.keys():
            self.file['comment'][...] = value
        else:
            self.file.create_dataset('comment', data=value)

    @property
    def n_baseline(self):
        if 'n_baseline' in self.file.keys():
            return self.file['n_baseline'][()]
        else:
            return 30

    @property
    def n_stim(self):
        if 'n_stim' in self.file.keys():
            return self.file['n_stim'][()]
        else:
            return 30

    @property
    def n_recovery(self):
        if 'n_recovery' in self.file.keys():
            return self.file['n_recovery'][()]
        else:
            return 20

    @property
    def dt(self):
        if 'dt' in self.file.keys():
            return self.file['dt'][()]
        else:
            return .1

    @property
    def cov(self):
        try:
            return self.file['df']['cov'][()]
        except KeyError:
            return None

    @property
    def max_project(self):
        return self.file['df']['max_project'][()]
    
    @property
    def response(self):
        return self.file['df']['response'][()]

    @property
    def stack(self):
        return self.file['df']['stack'][()]

    @property
    def avg_stack(self):
        return self.stack

    def _make_frame(self, t):
        full_time = np.linspace(0, self._duration, self._movie_stack.shape[2])
        ix = np.intp(np.argmin(np.abs(full_time - t)))
        c_frame = self._movie_stack[..., ix]
        c_frame = gauss_filt(c_frame, 5)

        return viridis(c_frame, alpha=None, bytes=True)[..., :3]

    @property
    def norm_stack(self):
        if self._norm_stack is None:
            self._norm_stack = normalize_stack(self.stack, self.n_baseline)
        return self._norm_stack

    @property
    def resp_map(self):
        if 'resp_map' not in self.file['df']:
            self.resp_mapping()
        resp = self.file['df']['resp_map'][()]
        return resp

    def export_movie(self, duration=10):
        try:
            self._duration = duration
            self._movie_stack = self.norm_stack.copy()
            self._movie_stack = np.clip(self._movie_stack, 0, np.percentile(self._movie_stack, 99))
            # self._movie_stack = np.clip(self._movie_stack, 0, 0.01)
            self._movie_stack = (self._movie_stack - self._movie_stack.min()) / (self._movie_stack.max() - self._movie_stack.min())
            animation = mpy.VideoClip(self._make_frame, duration=duration)
        except NameError as e:
            print(f'Name Error: {e}')
            return
        name = self.path.name[:-3]
        animation.write_videofile((self.path.parent / f'movie_{name}.mp4').as_posix(), fps=30)

    def export_response(self):
        # resp = np.mean(self.norm_stack[..., self.n_baseline:(self.n_baseline+30)], 2)
        if 'resp_map' not in self.file['df']:
            self.resp_mapping()
        resp = self.file['df']['resp_map'][()]
        # resp = np.clip(resp, 0, 2/100)
        # resp = exposure.equalize_hist(resp)
        resp = viridis(gauss_filt(resp, 3))[..., :3]
        resp[self.max_project <= 0, :] = 0
        # resp = 255 * (resp - resp.min()) / (resp.max() - resp.min())
        resp = img_to_uint8(resp)
        # resp = np.uint8(resp)
        imsave(self.path.parent / f'response_{self.path.name}.png', resp)

    def resp_mapping(self):
        r, df = resp_map(self.norm_stack, self.n_baseline, self.n_stim)
        df_grp = self.file['df']
        try:
            df_grp['resp_map'][...] = r
            df_grp['avg_df'][...] = df.mean(0)
            df_grp['max_df'][...] = df.max(1).mean()
            df_grp['area'][...] = np.sum(r > 0)
        except KeyError:
            df_grp.create_dataset('resp_map', data=r)
            df_grp.create_dataset('avg_df', data=df.mean(0))
            df_grp.create_dataset('max_df', data=df.max(1).mean())
            df_grp.create_dataset('area', data=np.sum(r > 0))

    def export_timecourse(self, xs, xe, ys, ye, dt=0.1):
        """" Make a nice figure of the time course of the given ROI and saves it"""
        fig = figure()
        ax = fig.add_subplot(1, 1, 1)
        t = np.arange(0, self.stack.shape[-1] * dt, dt)
        df = self.norm_stack[xs:xe, ys:ye, :]
        ax.plot(t, 100 * df.mean((0, 1)), linewidth=3)

        ax.set_xlabel(TIME_LABEL, fontsize=14, fontname=DEJAVU)
        ax.set_ylabel('$\Delta$ F / F', fontsize=14, fontname=DEJAVU)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        y_bot = ax.get_ylim()[0]
        y_top = ax.get_ylim()[1] * .09
        stim_rect = Rectangle((self.n_baseline * dt, y_bot), self.n_stim * dt, y_top,
                              facecolor='darkred')
        ax.add_patch(stim_rect)
        ax.text((self.n_baseline*2+self.n_stim)/2*dt, y_bot + y_top/2, 'Stimulation', color='w',
                verticalalignment='center', horizontalalignment='center')
        for xtick, ytick in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            xtick.set_fontname(DEJAVU)
            ytick.set_fontname(DEJAVU)

        fig.tight_layout()
        save_path = self.get_file(self.path, 'svg')
        fig.savefig(save_path.as_posix())
        return ax

    @staticmethod
    def get_file(path: Path, ext: str) -> Path:
        """
        Create a filename that does not already exists by appending a number to a given file name

        Parameters
        ----------
        path: Path
            Example of desired path, or original data path for example
        ext: str
            Required extension (without any dot)

        Returns
        -------
        save_path: Path
            Path to save to

        """
        num = 0
        save_path = path.parent / (path.stem + f'_{num}.{ext}')
        while save_path.is_file():
            num += 1
            save_path = path.parent / (path.stem + f'_{num}.{ext}')
        return save_path

    def export_tc_to_csv(self, xs, xe, ys, ye, dt=0.1):
        """
        Export the time course of a given ROI (eg from the GUI) to a CSV file

        Parameters
        ----------
        xs: int
            Start or ROI in x
        xe: int
            End of ROI in x
        ys: int
        ye: int
        dt: float
            Time spacing

        Returns
        -------
        save_path: str
            Path to which the export has been made
        """
        df = self.norm_stack[xs:xe, ys:ye, :]
        df_avg = df.mean((0, 1))
        df_std = df.std((0, 1))
        s_path = self.path.parent / (self.path.stem + '_timecourse.csv')
        save_path = self.get_file(s_path, 'csv')
        np.savetxt(save_path,
                   np.vstack((np.arange(1, df_avg.shape[0]+1, dtype=np.intp), df_avg, df_std)).T,
                   ["%d", "%.4f", "%.4f"], delimiter=",", header='frame, average, std')
        return save_path

    def export_resp_prm(self):
        """
        Export response parameters (average time course, maximal df and area) to a CSV file

        """
        grp = self.file['df']
        if 'resp_map' not in grp.keys():
            self.resp_mapping()
        # grp['resp_map'][...] = r
        # DONE: Export all responding pixels time courses in a CSV file
        resp = grp['response'][()]
        avg_df = grp['avg_df'][()]
        max_df = grp['max_df'][()]
        area_df = grp['area'][()]
        stack = grp['stack'][()]
        resp_pixels = stack[resp > 0, ...]
        s_path = self.path.parent / (self.path.stem + '_response.csv')
        all_path = self.path.parent / (self.path.stem + '_all_pixels.csv')
        save_path = self.get_file(s_path, 'csv')
        save_all_path = self.get_file(all_path, 'csv')
        np.savetxt(save_path,
                   np.vstack((np.arange(1, avg_df.shape[0] + 1, dtype=np.intp), avg_df)).T,
                   ["%d", "%.4f"], delimiter=",", header='frame, average',
                   footer=f'Maximum: {max_df} - Area: {area_df} pixels')
        np.savetxt(save_all_path, resp_pixels, '%.4f', delimiter=',')


def normalize_stack(stack, n_baseline=30):
    # Global average
    y = np.nanmean(stack, (0, 1))[1:n_baseline]
    y_min, y_max = y.min(), y.max()
    # Exponential fit during baseline
    t = np.arange(1, n_baseline)
    z = 1 + (y - y_min) / (y_max - y_min)
    p = np.polyfit(t, np.log(z), 1)
    # Modeled decay
    full_t = np.arange(stack.shape[2])
    decay = np.exp(p[1]) * np.exp(full_t * p[0])
    # Renormalized
    decay = (decay - 1) * (y_max - y_min) + y_min
    norm_stack = stack - decay

    return norm_stack


def find_resp(avg_stack, n_baseline=30, pvalue=0.05):
    t = np.arange(0, avg_stack.shape[2])
    sw = stim(t, t_on=n_baseline, tau_on=5, tau_off=15)
    reg = np.hstack((np.zeros(n_baseline), sw[n_baseline:]))
    reg = (reg - reg.min()) / (reg.max() - reg.min())
    reg *= avg_stack[..., n_baseline:].max()
    resp = np.zeros((avg_stack.shape[0], avg_stack.shape[1]))
    for row, r_slice in tqdm(enumerate(avg_stack)):
        for col, c_slice in enumerate(r_slice):
            # cs = (c_slice - c_slice.min()) / (c_slice.max() - c_slice.min())
            # cs -= cs[:n_baseline].mean()
            # cs /= cs[n_baseline:].max()
            # nreg = (reg - reg.min()) / (reg.max() - reg.min())
            r, p = pearsonr(c_slice[1:-1], reg[1:-1])
            if p < pvalue/(resp.shape[0]*resp.shape[1]) or p < pvalue:
                resp[row, col] = r
    f_resp = med_filt(resp, disk(3))
    resp[f_resp == 0] = 0

    return resp


def resp_map(norm_stack, n_baseline=30, n_stim=30):
    """
    Compute a smooth response image

    Parameters
    ----------
    norm_stack: 3D np array
    n_baseline: int
    n_stim: int

    Returns
    -------
    im_resp: numpy array
        Response map
    df: numpy array
        Signal of all responding pixels
    """
    # r = norm_stack[..., n_baseline:n_baseline+n_stim].mean(2)
    r = find_resp(norm_stack)
    # Keep only the top 5% pixels
    z = r > np.percentile(r, 95)
    im_resp = np.zeros(r.shape)
    im_resp[z] = r[z]
    mask = np.ones(r.shape)
    # mask[100:-100,100:-100] = 1
    im_resp = im_resp * mask
    im_resp = gauss_filt(im_resp, 2)
    im_resp = clean_response(im_resp)
    df = norm_stack[im_resp > 0, :]

    return im_resp, df


def id_sources(n_baseline=30, avg_stack=None, resp=None):
    if avg_stack is not None:
        x, y = np.nonzero(clean_response(find_resp(avg_stack, n_baseline)))
    elif resp is not None:
        x, y = np.nonzero(resp)
    else:
        raise ValueError('No suitable data provided')
    X = np.vstack((x, y)).transpose()
    try:
        clf = mixture.BayesianGaussianMixture(n_components=2)
        clf.fit(X)
        centers = clf.means_.transpose()[::-1, :]
        cov = clf.covariances_.transpose()[::-1, :]
        labels = clf.predict(X)
    except ValueError:
        # No response
        centers = np.zeros((2, 2)) * np.nan
        cov = np.zeros((2, 2)) * np.nan
        labels = [np.nan]

    return labels, centers, cov


def overlay(ref, avg_stack=None, resp=None, cm=YlOrRd):
    brain = np.float32(ref)
    brain = img_to_uint8(brain)
    brain = grey2rgb(brain)
    if resp is None and avg_stack is not None:
        resp = find_resp(avg_stack)
    elif resp is None and avg_stack is None:
        raise ValueError('No suitable data provided')
    fresp = gauss_filt(resp, 2)
    rgb_resp = cm(fresp)[..., :3]
    rgb_resp = gauss_filt(rgb_resp, 2)
    rgb_resp = img_to_uint8(rgb_resp)
    rgb_resp[resp <= 0, :] = 0
    overlaid = np.uint8(.6 * brain + .4 * rgb_resp)

    return overlaid


def multi_analysis(path: Union[str, Path], pattern=ALL_PNG):
    path = Path(path)
    folders = set()
    for file in path.glob(f'**/{pattern}'):
        if file.parent == path:
            continue
        if pattern == '*.tif':
            folders.add(file.parent)
        elif pattern == ALL_PNG:
            folders.add(file.parent.parent)

    for folder in tqdm(folders, desc='Analysing all folders'):
        # exts = Counter([f.name.split('.')[-1] for f in folder.iterdir() if f.is_file()])
        # most_common = exts.most_common(1)[0][0]
        try:
            print(folder)
            session = Intrinsic(folder, pattern, binning=2)
            session.save_analysis()
        except ValueError:
            warn('Probably not a valid trial folder')
    return True


def clean_response(resp):
    resp2 = med_filt(resp.copy(), disk(3)) > 0
    l_im = label(resp2)
    obj = regionprops(l_im)
    f_obj = [o for o in obj if o.area > 100]
    f_resp = np.zeros(resp.shape)
    for o in f_obj:
        f_resp[o.coords[:, 0], o.coords[:, 1]] = resp[o.coords[:, 0], o.coords[:, 1]]
    return f_resp


def category_stim(path: Union[str, Path]):
    path = Path(path)
    cat = ['whisker', 'trunk', 'visual', 'unknown']
    datastores = path.glob('**/*.h5')
    experiments = {}
    for f in datastores:
        exp_folder = f.parent.as_posix()
        exp_folder_l = f.parent.as_posix().lower()
        type_exp = list(filter(lambda x: x in exp_folder_l, cat))
        if type_exp:
            type_exp = type_exp[0]
        else:
            type_exp = 'unknown'
        experiments[exp_folder] = type_exp

    df_exp = pd.DataFrame({'experiment': list(experiments.keys()), 'stim': list(experiments.values())})

    return df_exp


def list_h5(path, strain='shank', show=False):
    path = Path(path)
    l_dfs = []
    t = np.arange(0, 8, .1)
    w = np.ones(len(t))
    w[25:] = 5
    for f in path.glob('*.h5'):
        S = Session(f)
        comment = S.comment.lower()
        if strain not in comment and 'wt' not in comment:
            continue
        if 'wt' in comment:
            genotype = 'wt'
        elif '+/-' in comment:
            genotype = '+/-'
        elif 'ko' in comment:
            genotype = 'ko'
        else:
            genotype = 'na'
        if 'bms' in comment:
            treatment = 'bms'
        elif 'veh' in comment:
            treatment = 'veh'
        else:
            treatment = 'veh'
        grp = S.file['df']
        if 'resp_map' not in grp.keys():
            S.resp_mapping()
        signal = grp['avg_df'][()][:80]
        if show:
            plt.figure()
            plt.imshow(grp['resp_map'])
            plt.title(f'{comment}  {S.path.name}')
        if signal[0] < 0:
            signal[0] = signal[1]
        peak = signal.max()
        hw = fwhm(signal, t, peak, 1, w=w)
        # plt.title(S.path)
        l_dfs.append(pd.DataFrame({'path': f.as_posix(), 'strain': strain, 'genotype': genotype,
                                   'area': grp['area'][()], 'max_df': grp['max_df'][()],
                                   'fwhm': hw, 'peak': peak*100,
                                   'avg_df': [100 * signal], 'treatment': treatment}))
        S.close()

    df = pd.concat(l_dfs)
    df['area'] = df['area'] * (6.5 / 1.6 / 1.6) ** 2  # Pixel size is 6.5µm, zoom is 1.6^2
    df['max_df'] = df['max_df'] * 100
    return df


def figure_size_resp(df):
    """
    Plot a few parameters relative to the response in flavo imaging

    Parameters
    ----------
    df: pandas dataframe
        As returned by list_h5

    Returns
    -------

    """
    sns.set_style('ticks')
    gs = GridSpec(2, 3)
    fig = plt.figure(figsize=(7, 8))
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
           fig.add_subplot(gs[1, :])]
    # fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    # axs = axs.reshape(-1)

    sns.boxplot('genotype', 'area', data=df, ax=axs[0], order=('wt', 'ko', '+/-'))
    axs[0].set_ylim((0, 800000))
    axs[0].set_ylabel('Responsive area in µm²')
    sns.boxplot('genotype', 'max_df', data=df, ax=axs[1], order=('wt', 'ko', '+/-'))
    axs[1].set_ylabel('Average peak response amplitude (%)')
    axs[1].set_ylim((0, 3))
    sns.boxplot('genotype', 'fwhm', data=df, ax=axs[2], order=('wt', 'ko', '+/-'))
    avg_df = df.groupby('genotype').avg_df.apply(lambda x: np.vstack(x.as_matrix()))
    t = np.arange(-3, 5, .1)
    for g in ('wt', 'ko', '+/-'):
        try:
            mean_df = avg_df[g].mean(0)
            # mean_df[mean_df > 0.7] = 0
            axs[3].plot(t, mean_df, label=g, linewidth=2)
        except KeyError:
            pass
    axs[3].legend()
    axs[3].set_xlabel(TIME_LABEL)
    axs[3].set_ylabel('Average $\Delta$ F / F (%)')
    fig.tight_layout()
    fig.savefig('Intrinsic/figure/responses.png')
    fig.savefig('Intrinsic/figure/responses.svg')
    with open('Intrinsic/figure/stats.txt', 'w') as f:
        f.write('Mann-Whitney U-test\n\n')
        for g1, g2 in combinations(('wt', 'ko', '+/-'), 2):
            f.write(f'+ {g1} vs {g2}:\n')
            pval = mannwhitneyu(df.area[df.genotype == g1], df.area[df.genotype == g2]).pvalue
            f.write(f'\tArea comparison {g1} vs {g2}: {pval:.3f}\n')
            pval = mannwhitneyu(df.max_df[df.genotype == g1], df.max_df[df.genotype == g2]).pvalue
            f.write(f'\tAmplitude comparison {g1} vs {g2}: {pval:.3f}\n')
            pval = mannwhitneyu(df.fwhm[df.genotype == g1], df.fwhm[df.genotype == g2]).pvalue
            f.write(f'\tFull width at half maximum comparison {g1} vs {g2}: {pval:.3f}\n')


def figure_size_resp_bms(df):
    """
    Plot a few parameters relative to the response in flavo imaging
    Special case for treatment
    Parameters
    ----------
    df: pandas dataframe
        As returned by list_h5

    Returns
    -------

    """
    sns.set_style('ticks')
    gs = GridSpec(2, 3)
    fig = plt.figure(figsize=(7, 8))
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
           fig.add_subplot(gs[1, :])]
    # fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    # axs = axs.reshape(-1)

    sns.boxplot('genotype', 'area', hue='treatment', data=df, ax=axs[0], order=('wt', 'ko'), hue_order=('veh', 'bms'))
    axs[0].set_ylim((0, 2000000))
    axs[0].set_ylabel('Responsive area in µm²')
    sns.boxplot('genotype', 'max_df', hue='treatment', data=df, ax=axs[1], order=('wt', 'ko'), hue_order=('veh', 'bms'))
    axs[1].set_ylabel('Average peak response amplitude (%)')
    axs[1].set_ylim((0, 3.5))
    sns.boxplot('genotype', 'fwhm', hue='treatment', data=df, ax=axs[2], order=('wt', 'ko'), hue_order=('veh', 'bms'))
    gp = df.groupby(('genotype', 'treatment'))
    t = np.arange(-3, 5, .1)
    for g in product(('wt', 'ko'), ('veh', 'bms')):
        try:
            avg_df = np.vstack(gp.get_group(g).avg_df.as_matrix())
            mean_df = avg_df.mean(0)
            # mean_df[mean_df > 0.7] = 0
            axs[3].plot(t, mean_df, label=g, linewidth=2)
        except KeyError:
            pass
    axs[3].legend()
    axs[3].set_xlabel(TIME_LABEL)
    axs[3].set_ylabel('Average $\Delta$ F / F (%)')
    fig.tight_layout()
    fig.savefig('Intrinsic/figure/responses.png')
    fig.savefig('Intrinsic/figure/responses.svg')
    with open('Intrinsic/figure/stats.txt', 'w') as f:
        f.write('Mann-Whitney U-test\n\n')
        for g1, g2 in combinations(product(('wt', 'ko'), ('veh', 'bms')), 2):
            f.write(f'+ {g1} vs {g2}:\n')
            pval = mannwhitneyu(df.area[df.genotype == g1], df.area[df.genotype == g2]).pvalue
            f.write(f'\tArea comparison {g1} vs {g2}: {pval:.3f}\n')
            pval = mannwhitneyu(df.max_df[df.genotype == g1], df.max_df[df.genotype == g2]).pvalue
            f.write(f'\tAmplitude comparison {g1} vs {g2}: {pval:.3f}\n')
            pval = mannwhitneyu(df.fwhm[df.genotype == g1], df.fwhm[df.genotype == g2]).pvalue
            f.write(f'\tFull width at half maximum comparison {g1} vs {g2}: {pval:.3f}\n')


def find_roots(func, a, b, step=1e-3, roots=None):
    """
    Find all the roots of scalar function `func` that are between `a` and `b`

    Parameters
    ----------
    func: function
    a: float
        Left bound
    b: float
        Right bound
    step: float
        Size of the window to look for a sign change of `func`
    roots: None

    Returns
    -------
    roots: list
        List of roots
    """
    if roots is None:
        roots = []
    if len(roots) > 100:
        return roots
    if func(a) == 0:
        roots.append(a)
        find_roots(func, a+1e-3, b, roots)
    if func(b) == 0:
        roots.append(b)
        find_roots(func, a, b-1e-3, roots)
    # n_steps = 1000
    # step = 1./n_steps
    n_steps = int(1./step)
    win_size = b - a
    for s in np.linspace(a+step, b-step, n_steps):
        if func(s) * func(s+step*win_size) >= 0:
            continue
        r = brentq(func, s, s+step*win_size)
        roots.append(r)

    return roots


def fwhm(sweep, t, amp, min_dt=10, w=None):
    c = find_cross(t, sweep - amp/2, min_dt=min_dt, w=w)
    return np.diff(c)


def find_cross(t, trace, min_dt=50, w=None):
    if w is None:
        w = np.ones(len(t))
    sp_data = UnivariateSpline(t, trace, k=5, s=0, w=w)
    crosses = np.array(find_roots(sp_data, 3, t.max()))
    gc = np.ones((len(crosses), ), dtype=np.bool)
    gc[1:] = np.diff(crosses) > min_dt
    g_crosses = crosses[gc]
    plt.figure()
    plt.plot(t, trace)
    plt.plot(t, sp_data(t))
    if len(g_crosses) > 2:
        start = g_crosses.min()
        g_crosses = np.hstack([start, g_crosses[g_crosses-start > min_dt].min()])
    if len(g_crosses) == 0:
        g_crosses = np.zeros((2,)) + np.nan
    if len(g_crosses) == 1 and g_crosses[0] < 4:
        g_crosses = np.hstack((g_crosses, [t.max()]))
    elif len(g_crosses) == 1:
        g_crosses = np.hstack(([3], g_crosses))
    plt.scatter(g_crosses, sp_data(g_crosses))
    return g_crosses


if __name__ == '__main__':
    # p = Path('D:\\20180222')
    # p = Path('/media/remi/LaCie/Guillaume - Flavo Xavier')
    # multi_analysis(p, ALL_PNG)

    # I = Intrinsic('/home/remi/Programming/EPhys/Intrinsic/data/20180222_1729_0', binning=3)
    # I.average_trials()
    # I.save_analysis()
#    plt.ion()
    pass
