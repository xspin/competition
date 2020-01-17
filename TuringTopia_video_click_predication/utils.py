import random
import time
import pandas as pd
from collections import defaultdict
import gc
import numpy as np
import os, sys
import pickle
import logging
from sklearn import preprocessing
logging.basicConfig(level=logging.INFO,
                format='[%(asctime)s] %(levelname)s: %(message)s', 
                datefmt='%Y-%m-%d %H:%M:%S')

def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    gc.collect()
    logging.info('Reduced memory: {:.2f} Mb -> {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def sample_dataset(src_path, dst_path, rate):
    logging.info(f'Sampling from {src_path} to {dst_path} ...')
    src_lines = 0
    dst_lines = 0
    with open(dst_path, 'w') as dstf:
        with open(src_path, 'r') as srcf:
            dstf.write(srcf.readline())
            for line in srcf:
                src_lines += 1
                if(random.random()<=0.1):
                    dstf.write(line)
                    dst_lines += 1
    logging.info(f'Sampled: {dst_lines}/{src_lines} = {dst_lines/src_lines}') 

def load_data(path):
    """ Load train/test csv data to pandas.Dataframe
    """
    df = pd.read_csv(path)
    return df

def merge(df, df_app, df_user):
    df = pd.merge(df, df_app, how='left', on='deviceid')
    df = pd.merge(df, df_user, how='left', on=['deviceid', 'guid'])
    return df

def get_apps(series_apps):
    appcount = defaultdict(int)
    for apps in series_apps:
        for app in apps[1:-1].split(): appcount[app] += 1
    for app in appcount: appcount[app] /= series_apps.shape[0]
    return sorted(appcount.items(), key=lambda x:x[1], reverse=True)

def get_app_cols(series_apps, applist):
    tmp = {app:[0]*series_apps.shape[0] for app in applist}
    for i, s in enumerate(series_apps):
        if(pd.isnull(s)): continue
        for app in s[1:-1].split(): 
            if app in tmp: tmp[app][i] = 1
    return pd.DataFrame.from_dict(tmp)

def get_tags(series_tags):
    tagcount = defaultdict(int)
    for tags in series_tags:
        if(pd.isnull(tags)): continue
        for t in tags.split('|'):
            try:
                tag, prob = t.split(':')
            except:
                tag = t
                prob = 0.
            tagcount[tag] += float(prob)
    for tag in tagcount: tagcount[tag] /= series_tags.shape[0]
    tags = sorted(tagcount.items(), key=lambda x:x[1], reverse=True)
    if 'TOP' in tags[0][0]: tags = tags[1:]
    return tags

def get_tag_cols(series_tags, taglist, prefix='tag_'):
    if taglist==[]: return None
    tmap = dict(zip(taglist, map(lambda i:prefix+str(i), range(len(taglist)))))
    tmp = {tmap[tag]:[0.0]*series_tags.shape[0] for tag in taglist}
    for i, s in enumerate(series_tags):
        if(pd.isnull(s)): continue
        for t in s.split('|'):
            try:
                tag, prob = t.split(':')
            except:
                tag = t
                prob = 0.
            if tag in tmp:
                tmp[tmap[tag]][i] = float(prob)
    return pd.DataFrame.from_dict(tmp).astype('float32')

def get_click_df(train_df):
    click_df = train_df[train_df['target'] == 1].sort_values('timestamp').reset_index(drop=True)
    click_df['exposure_click_gap'] = click_df['timestamp'] - click_df['ts']
    click_df = click_df[click_df['exposure_click_gap'] >= 0].reset_index(drop=True)
    click_df['date'] = pd.to_datetime(
        click_df['timestamp'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
    )
    click_df['day'] = click_df['date'].dt.day
#     click_df.loc[click_df['day'] == 7, 'day'] = 8
    return click_df

def extend_features(df, applist, taglist=[], outertaglist=[]):
    logging.info('Extending features ...')
    logging.info(f'apps: {applist}')
    if applist:
        tmp_apps = get_app_cols(df['applist'], applist)
        df = pd.concat([df, tmp_apps], axis=1)
    del df['applist']
    logging.info(f'tags: {taglist}')
    if taglist:
        tmp = get_tag_cols(df['tag'], taglist)
        df = pd.concat([df, tmp], axis=1)
    del df['tag']
    logging.info(f'outertags: {outertaglist}')
    if outertaglist:
        tmp = get_tag_cols(df['outertag'], outertaglist, 'otag_')
        df = pd.concat([df, tmp], axis=1)
    del df['outertag']
    gc.collect()
    return df

def preprocess(df, cate_cols, del_cols=[], ignores=[]):
    logging.info('Preprocessing ...')
    df['lng_lat'] = df['lng'].map(lambda x:'{:.2f}'.format(x)) + '_' \
                    + df['lat'].map(lambda x:'{:.2f}'.format(x))
    df['date'] = pd.to_datetime(
        df['ts'].apply(lambda x: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(x / 1000)))
    )
    df['weekday'] = df['date'].dt.weekday
    # df['month'] = df['date'].dt.month
    # df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    # df['minute'] = df['date'].dt.minute
    for f in del_cols: del df[f]
    # process sparse features
    for f in cate_cols: 
        if f in ignores: continue
        logging.info(f'{f}: {df[f].nunique()}')
        map_dict = dict(zip(df[f].unique(), range(1,1+df[f].nunique())))
        df[f] = df[f].map(map_dict).fillna(0).astype('int32')
    # process dense features
    for f in df.columns:
        if f in ignores: continue
        df[f] = df[f].fillna(0)
    df['lng'] /= 360
    df['lat'] /= 360
    for col in ['level', 'personidentification', 'followscore', 'personalscore']:
        df[col] /= df[col].max()
    gc.collect()
    df = reduce_mem(df)
    return df

def get_dataset_df(data_dir, debug=False):
    dataset_df_dir = os.path.join(data_dir, 'data.pickle')
    cate_list_dir = os.path.join(data_dir, 'cate.pickle')
    if debug:
        logging.info('============ [Debug mode] ===========')
        dataset_df_dir = os.path.join(data_dir, 'data_samples.pickle')
        cate_list_dir = os.path.join(data_dir, 'cate_samples.pickle')
    if os.path.exists(dataset_df_dir): 
        logging.info(f'Find and load pickle files from {dataset_df_dir}')
        with open(dataset_df_dir, 'rb') as datasetfp, \
            open(cate_list_dir, 'rb') as catefp:
            df = pickle.load(datasetfp)
            num_train = df['target'].index[df['target'].apply(np.isnan)][0]
            return df, pickle.load(catefp), num_train

    app_path = os.path.join(data_dir, 'app.csv')
    user_path = os.path.join(data_dir, 'user.csv')
    sample_path = os.path.join(data_dir, 'sample.csv')
    if debug:
        data_dir = 'dataset/samples'
    test_path = os.path.join(data_dir, 'test.csv')
    train_path = os.path.join(data_dir, 'train.csv')
    logging.info('Loading app/user dataset ...')    
    df_app = pd.read_csv(app_path)
    logging.info(f'app: {df_app.shape}')
    df_app.drop_duplicates(subset='deviceid', inplace=True)
    logging.info(f'app: {df_app.shape}')
    df_user = pd.read_csv(user_path)
    logging.info(f'user: {df_user.shape}')
    df_user.drop_duplicates(subset=['deviceid', 'guid'], inplace=True)
    logging.info(f'user: {df_user.shape}')
    for col in ['deviceid', 'guid']: df_user[col] = df_user[col].fillna('*')
    logging.info('Loading test/train dataset ...')    
    df_test = load_data(test_path)
    logging.info(f'test: {df_test.shape}')
    df_train = load_data(train_path)
    logging.info(f'train: {df_train.shape}')
    del df_train['timestamp']

    logging.info('Concating train & test ...')
    num_train = df_train.shape[0]
    df = pd.concat([df_train, df_test], axis=0, sort=False)
    df.reset_index(drop=True, inplace=True)
    del df_train, df_test
    logging.info(f'dataset: {df.shape}')

    gc.collect()
    logging.info('Merging ...')
    df = merge(df, df_app, df_user)
    # logging.info(' test', df_test.shape)
    # df_train = merge(df_train, df_app, df_user)
    # logging.info(' train', df_train.shape)

    n_apps = 10
    n_tags = 10 
    n_outertags = 10
    logging.info(f'Getting the top {n_tags} apps list ...')
    apps = get_apps(df_app['applist'])
    applist = list(map(lambda x:x[0], apps[:n_apps]))
    buf = '  '
    for app in apps[:n_apps]:
        buf += '{}:{:.2f} '.format(*app)
    logging.info(buf)
    logging.info(f'Getting the top {n_tags} tags list ...')
    tags = get_tags(df_user['tag'])
    taglist = list(map(lambda x:x[0], tags[:n_tags]))
    buf = '  '
    for tag in tags[:n_tags]:
        buf += '{}:{:.2f} '.format(*tag)
    logging.info(buf)
    logging.info(f'Getting the top {n_tags} outertags list ...')
    outertags = get_tags(df_user['outertag'])
    outertaglist = list(map(lambda x:x[0], outertags[:n_outertags]))
    buf = '  '
    for tag in outertags[:n_outertags]:
        buf += '{}:{:.2f} '.format(*tag)
    logging.info(buf)

    logging.info('Extending dataset ...')
    df = extend_features(df, applist, taglist, outertaglist)

    cate_cols = [
        'newsid', 'guid', 'pos', 'app_version', 'device_vendor',
        'netmodel', 'osversion', 'device_version', 'gender', 
        'weekday', 'hour', 'lng_lat', 'deviceid'
    ]
    del_cols = ['id', 'date', 'ts']
    ignores = ['target']

    cate_cols.sort()

    df = preprocess(df, cate_cols, del_cols, ignores)

    logging.info(f'Dumping to {dataset_df_dir}, {cate_list_dir} ...')
    with open(dataset_df_dir, 'wb') as datasetfp: 
        pickle.dump(df, datasetfp)
    with open(cate_list_dir, 'wb') as catefp: 
        pickle.dump(cate_cols, catefp)
    logging.info('done')
    return df, cate_cols, num_train

def binary_embedding(n):
    d = int(np.ceil(np.log2(n)))
    def binary_list(x, d):
        s = bin(x)[2:]
        if len(s)<d: s = '0'*(d-len(s))+s
        return list(map(int, s))
    t =  np.array([binary_list(x, d) for x in range(n)]).astype('float32')
    return preprocessing.normalize(t, norm='l1').astype('float32')

def random_embedding(n):
    np.random.seed(2020)
    d = int(np.ceil(np.log2(n)))
    return preprocessing.normalize(np.random.randn(n, d), norm='l2').astype('float32')

def load_embedding(df, sparse_cols, data_dir, method='binary'):
    if sys.platform == 'linux':
        data_dir = '/root'
    embedding_path = os.path.join(data_dir, 'embedding_%s.pickle'%method)
    if os.path.exists(embedding_path):
        logging.info(f'Loading embedding from {embedding_path}')
        with open(embedding_path, 'rb') as fp:
            return pickle.load(fp)
    embed_func = binary_embedding if method=='binary' else random_embedding
    embedding = {col:embed_func(df[col].max()+1) for col in sparse_cols}
    logging.info(f'Dumping embedding to {embedding_path}')
    with open(embedding_path, 'wb') as fp:
        pickle.dump(embedding, fp)
    return embedding

def process_embedding(df, dense_cols, embedding):
    x = df[dense_cols].values.astype('float32')
    for col, embed in embedding.items():
        t = np.array([embed[int(v)] for v in df[col]], dtype='float32')
        x = np.concatenate((x, t), axis=1)
    return x

def gen_batch(xdata, batch_size, drop_last=True, one_epoch=False, shuffle=True):
    while True:
        if shuffle:
            np.random.shuffle(xdata)
        for i in range(0, len(xdata), batch_size):
            if drop_last and i+batch_size>len(xdata): break
            yield xdata[i:min(i+batch_size, len(xdata))]
        if one_epoch: break
def gen_batch2(data, batch_size, drop_last=True, one_epoch=False, shuffle=True):
    xdata, ydata = data
    idx = np.arange(len(xdata), dtype='int32')
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(xdata), batch_size):
            if drop_last and i+batch_size>len(xdata): break
            yield (xdata[idx[i:min(i+batch_size, len(xdata))]], ydata[idx[i:min(i+batch_size, len(xdata))]])
        if one_epoch: break

def sec2hms(sec):
    t = sec
    s, t = t%60, t//60
    m, t = t%60, t//60
    h, d = t%24, t//24
    if d > 0: return "{:.0f}d {:.0f}h {:.0f}m {:.0f}s".format(d,h,m,s)
    if h > 0: return "{:.0f}h {:.0f}m {:.0f}s".format(h,m,s)
    if m > 0: return "{:.0f}m {:.0f}s".format(m,s)
    return "{:.02f}s".format(s)

class Clock:
    def __init__(self, n_steps=None):
        self.tic()
        self.n_steps = n_steps
    def tic(self):
        self.start_time = time.time()
    def toc(self, step=None):
        cost = time.time() - self.start_time
        if step is None or self.n_steps is None: 
            return sec2hms(cost)
        else:
            step += 1
            return sec2hms(cost), sec2hms(cost*(self.n_steps-step)/step)

def gen_feat_dict(df, sparse_cols, ignore_cols=[]):
    feat_dict = {}
    tc = 0
    for col in df.columns:
        if col in ignore_cols:
            continue
        if col in sparse_cols:
            us = df[col].unique()
            feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
            tc += len(us)
        else:
            # map to a single index
            feat_dict[col] = tc
            tc += 1
    feat_dim = tc
    return feat_dict, feat_dim

def parse_dataset(df, feat_dict, sparse_cols, ignore_cols=[]):
    dfi = df.copy()
    dfv = dfi.copy()
    for col in dfi.columns:
        if col in ignore_cols:
            dfi.drop(col, axis=1, inplace=True)
            dfv.drop(col, axis=1, inplace=True)
            continue
        elif col in sparse_cols:
            dfi[col] = dfi[col].map(feat_dict[col])
            dfv[col] = 1.
        else:
            dfi[col] = feat_dict[col]

    # list of list of feature indices of each sample in the dataset
    Xi = dfi.values
    # list of list of feature values of each sample in the dataset
    Xv = dfv.values
    del dfi, dfv
    gc.collect()
    return Xi, Xv

if __name__ == "__main__":
    pass

