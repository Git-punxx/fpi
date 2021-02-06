from intrinsic.imaging import resp_map
import h5py
import numpy as np
from app_config import config_manager as mgr

def analyze(df_file):
    with h5py.File(df_file, 'a') as ds:
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






