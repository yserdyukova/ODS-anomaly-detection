#!$CONDA_PYTHON_EXE

import pandas as pd
import numpy as np
import itertools
import glob
import pandas as pd
import os

from datetime import date, datetime
from sklearn import preprocessing

def file_to_df(filename, time_list = []):    
    ''' Read files from directory and put to pandas dataframe. 
            Input:
                filename - path to files
                time_list - date fields
            Output:
                pd.DataFrame() 
                
    '''
    
    AllFiles = glob.glob(filename)    
    assert len(AllFiles) > 0, 'No files in directory'
   
    list_ = [pd.DataFrame()]
    for file_ in AllFiles:
        columns = pd.read_csv(file_, sep=',', nrows=0).columns.str.lower().str.strip()
        df = pd.read_csv(file_, ',', names=columns, parse_dates=time_list, dayfirst=True, skiprows=1)
        list_.append(df)        
    assert len(list_) > 0
    
    return pd.concat(list_, axis=0, ignore_index=True, sort=True)


def process_files(df, drop_columns=[], rename_columns={}):

    return df.drop(drop_columns, axis=1).rename(index=str, columns=rename_columns).applymap(lambda x: x.strip().lower() if type(x) == str else x).replace('none', np.NaN)        

        
def prepare_features(df, group_excl):

    columns = df.columns.drop(['snap_id','plan_name','max_utilization_limit', 
                         'mgmt_p1', 'parallel_degree_limit_p1', 'parallel_target_percentage','pxenq',
                         'sys_dbtime','sys_actsess_avg'
                              ],errors='ignore')
    

    return df.loc[~df.consumer_group.isin(group_excl),columns]


def sample_features(df, sample_period='H'):
    
    ''' Resample to smooze time series.
    
    '''

    return df.groupby(['host', 'consumer_group', pd.Grouper(key='time', freq=sample_period)]).quantile(0.75).reset_index()


def scale_features(df):
    
    ''' Scale values by every consumer_group and host.
    
    '''
    host_cons = [(host,group) for host,group in df.drop_duplicates(['host','consumer_group'])[['host','consumer_group']].values]

    scaler = preprocessing.MinMaxScaler()
    
    df_scaled = pd.DataFrame()
    for (host,consumer_group) in host_cons:
        df_sample = df[(df.host == host)&(df.consumer_group == consumer_group)].set_index(['host','consumer_group','time'])
        df_sample = pd.concat([df_sample.reset_index()[['host','consumer_group','time']],pd.DataFrame(scaler.fit_transform(df_sample),columns=df_sample.columns)],axis=1,sort=False)
        df_scaled = pd.concat([df_scaled,df_sample],sort=False)

    return df_scaled


if __name__ == "__main__":
    
    df_sysMetrics = process_files(file_to_df('input/GrpStat_OSGLOB*.dat'), drop_columns=['time'])
    df_grpMetrics = process_files(file_to_df('input/GrpStat_RGALL*.dat'), drop_columns=['t_beg'], rename_columns={"instance_number": "host", "usr_group": "consumer_group"})
    df_directives = process_files(file_to_df('input/GrpStat_Directives*.dat', time_list=['begin_time']), rename_columns={"instance_number": "host", "begin_time": "time"})

    df = df_grpMetrics.set_index(['host','snap_id','consumer_group']).join(df_directives.set_index(['host','snap_id','consumer_group'])).reset_index().merge(df_sysMetrics,on=['host','snap_id'])
        
    df = prepare_features(df, group_excl=['other_groups','ods2exa_group'])
    df = sample_features(df, sample_period='3H')
    df = scale_features(df)
    
    out_name = 'rm_features.csv'
    out_dir = 'clear_data'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    full_out_name = os.path.join(out_dir, out_name)
    
    df.to_csv(full_out_name, ';', index=False)

