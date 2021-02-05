from intrinsic.imaging import resp_map
import h5py
import numpy as np

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

