from modified_intrinsic.imaging import resp_map
from modified_intrinsic.imaging import Intrinsic, ReducedStack, Session
from skimage.measure import block_reduce
import h5py
import numpy as np
from app_config import config_manager as mgr
from app_config import LOG_FILE
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import time
import concurrent.futures as fut
import time
import os
from pathlib import Path
import sys
import logging

class RawAnalysisController:
    def __init__(self, root_folder = None):
        self.root = root_folder
        self.trial_tree = {}
        self._assemble_tree()

    def _assemble_tree(self):
        trial_folders = [folder for folder in os.listdir(self.root) if 'Trial_' in folder]
        for folder in trial_folders:
            folder_path = os.path.join(self.root, folder)
            images = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if 'img_' in img]
            images.sort(key = lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            self.trial_tree[folder] = images
        return self.trial_tree

    def set_root(self, path):
        if not os.path.exists(path):
            raise ValueError('Path is not valid')
        self.root = path
        self._assemble_tree()
        print('Tree structure updated')


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


def monkeypatck(func):
    def deco(*args, **kwargs):
        return get_pic(*args, **kwargs)
    return deco

class StrategyStack(ReducedStack):
    def __init__(self, path, pattern, binning = 3, strategy = 'duplicate'):
        super().__init__(path, pattern, binning)
        self.strategy = strategy
        self.previous_im = None

    def get_pic(self, im):
        try:
            pic = super().get_pic(im)
            pic = block_reduce(pic, (self.binning, self.binning), np.mean)
            c_avg = pic.mean()
            if self._previous_avg is not None:
                ratio = c_avg / self._previous_avg
                if ratio > 1.5 or ratio < .7:
                    pic = pic / ratio
                    self._previous_avg = c_avg / ratio
            self._previous_avg = pic.mean()
            self.previous_im = pic
            return pic
        except Exception as e:
            print(e)
            if self.strategy == 'duplicate':
                return self.previous_im
            elif self.strategy == 'average':
                raise
            else:
                raise


class ThreadedIntrinsic(Intrinsic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def complete_analysis(self):
        # path to datastore: self.save_path
        print('Beginning baseline analysis')
        self.save_analysis()
        print('Completing analysis')
        analyze(self.save_path)
        #s = Session(self.save_path)
        #print('Exporting paremeters')
        #s.export_resp_prm()


    def mean_baseline(self, stack):
        sys.stdout.flush()
        s = stack[:self.n_baseline].mean(0)
        return s

    def compute_baselines(self):
        print('Computing baselines')
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.mean_baseline, stack) for stack in self.stacks]

        # self.l_base = [s[:self.n_baseline].mean(0)
                       #for s in tqdm(self.stacks, desc='Computing baseline')]
        self.l_base = [f.result() for f in futures]
        self.baseline = np.mean(self.l_base, 0)

def analyze(df_file):
    with h5py.File(df_file, 'a') as ds:
        print(f'Saving to {df_file}...')
        stack = ds['df']['stack'][()]
        r, df = resp_map(stack)
        df_grp = ds['df']
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



def completion_report(exp_path):
    STAGE = 0
    with h5py.File(exp_path, 'r') as df:
        if 'stack' in df['df'].keys():
            STAGE += 1
        if 'resp_map' in df['df'].keys():
            STAGE += 1
        if 'roi' in df['df'].keys():
            STAGE += 1
    return STAGE

def completion_color(stage):
    return mgr._json['stage_color'][str(stage)]


def experiment_statistics(exp_list):
    # TODO Create a report for the whole of the experiments
    pass



'''
What we need to do:
Given a path:
1. Descend into the child directories and find which one contains tiff files
2. Collect the directories that contain tiff files
3. Run analysis in every one of them in parallel
'''
logging.basicConfig(level=logging.DEBUG, filename=LOG_FILE, format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s', filemode='w', datefmt='%m-%d %H:%M')
logger = logging.getLogger('TIFF')

def tiff_analysis(path):
    logger.debug('[INFO] Analysis for path %s started', path)
    try:
        analysis = ThreadedIntrinsic(path, binning=1, pattern='*.tif')
        analysis.complete_analysis()
        logger.debug('[INFO] Analysis for path %s succeded', path)
    except Exception as e:
        logger.debug('[FAILURE] Analysis for path %s failed: %s', path, e)

def scan(path):
    p = Path(path)
    parent_dirs = set([str(p.parent) for p in p.glob('**/*.tif')])
    return parent_dirs

def do_analysis(path):
    dirs = scan(path)
    with fut.ThreadPoolExecutor(max_workers = 8) as executor:
        futures = [executor.submit(tiff_analysis, d) for d in dirs]
    [f.result() for f in futures]






