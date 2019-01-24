#!/usr/bin/env python3

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

    return df.drop(drop_columns, axis=1)        .rename(index=str, columns=rename_columns)        .applymap(lambda x: x.strip().lower() if type(x) == str else x)        .replace('none', np.NaN)        

        
def prepare_features(df, group_excl):

    columns = df.columns.drop(['snap_id','plan_name','max_utilization_limit', 
                         'mgmt_p1', 'parallel_degree_limit_p1', 'parallel_target_percentage','pxenq',
                         'sys_dbtime','sys_actsess_avg'],errors='ignore')

    return df.loc[~df.consumer_group.isin(group_excl),columns]


def sample_features(df, sample_period='H'):
    
    ''' Resample to smooze time series.
    
    '''
    #df=df.groupby(['host','consumer_group',df.index.date]).filter(lambda x: (x.index.min().hour==0)&(x.index.max().hour==23)).reset_index()
    #df_sample = df.set_index(['host','consumer_group']).groupby(level=['host','consumer_group']).apply(lambda c: c.set_index(pd.DatetimeIndex(c['time'])).resample(sample_period).agg(['quantile'])).reset_index().dropna()
    #df_sample.columns = ['time','host','consumer_group'] + ['_'.join(col).strip() for col in df_sample.columns.values if col[0] not in ['host','time','consumer_group']]

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


def dummy_features(df):
    
    ''' Get dummy features.
    
    '''
    
    holidays = {
    2018: {1:[1,2,3,4,5,6,7,8,13,14,20,21,27,28],2:[3,4,10,11,17,18,22,23,24,25],3:[3,4,7,8,9,10,11,17,18,24,25,31],4:[1,7,8,14,15,21,22,28,29,30],5:[1,2,5,6,8,9,12,13,19,20,26,27],6:[2,3,9,10,11,12,16,17,23,24,30],7:[1,7,8,14,15,21,22,28,29],8:[4,5,11,12,18,19,25,26],9:[1,2,8,9,15,16,22,23,29,30],10:[6,7,13,14,20,21,27,28],11:[3,4,5,10,11,17,18,24,25],12:[1,2,8,9,15,16,22,23,29,30,31]},
    2019: {1:[1,2,3,4,5,6,7,8,12,13,19,20,26,27],2:[2,3,9,10,16,17,22,23,24],3:[2,3,7,8,9,10,16,17,23,24,30,31],4:[6,7,13,14,20,21,27,28,30],5:[1,2,3,4,5,8,9,10,11,12,18,19,25,26],6:[1,2,8,9,11,12,15,16,22,23,29,30],7:[6,7,13,14,20,21,27,28],8:[3,4,10,11,17,18,24,25,31],9:[1,7,8,14,15,21,22,28,29],10:[5,6,12,13,19,20,26,27],11:[2,3,4,9,10,16,17,23,24,30],12:[1,7,8,14,15,21,22,28,29,31]},
    2020: {1:[1,2,3,4,5,6,7,8,11,12,18,19,25,26],2:[1,2,8,9,15,16,22,23,24,29],3:[1,7,8,9,14,15,21,22,28,29],4:[4,5,11,12,18,19,25,26,30],5:[1,2,3,8,9,10,11,16,17,23,24,30,31],6:[6,7,11,12,13,14,20,21,27,28],7:[4,5,11,12,18,19,25,26],8:[1,2,8,9,15,16,22,23,29,30],9:[5,6,12,13,19,20,26,27],10:[3,4,10,11,17,18,24,25,31],11:[1,3,4,7,8,14,15,21,22,28,29],12:[5,6,12,13,19,20,26,27,31]},
    2021: {1:[1,2,3,4,5,6,7,8,9,10,16,17,23,24,30,31],2:[6,7,13,14,20,21,22,23,27,28],3:[6,7,8,13,14,20,21,27,28],4:[3,4,10,11,17,18,24,25,30],5:[1,2,3,8,9,10,15,16,22,23,29,30],6:[5,6,11,12,13,14,19,20,26,27],7:[3,4,10,11,17,18,24,25,31],8:[1,7,8,14,15,21,22,28,29],9:[4,5,11,12,18,19,25,26],10:[2,3,9,10,16,17,23,24,30,31],11:[3,4,6,7,13,14,20,21,27,28],12:[4,5,11,12,18,19,25,26,31]}
    }
    holidays = pd.DataFrame(list(itertools.chain.from_iterable(list(itertools.chain.from_iterable([[[date(year,month,day) for day in days] for month,days in v.items()] for year,v in holidays.items()])))),columns=['date'])
    holidays.date = pd.to_datetime(holidays.date)
    holidays['is_holiday'] = 1
    
    df_dummy=df.copy(deep=True)
    
    df_dummy.loc[:, ('day_of_week')] = df_dummy.time.dt.dayofweek
    df_dummy.loc[:, ('hour_of_day')] = df_dummy.time.dt.hour
    #df_dummy.loc[:, ('is_weekend')] = df_dummy.time.dt.dayofweek.isin([5,6])*1
    df_dummy.loc[:, ('week_of_month')] = (df_dummy.time.dt.day - 1)//7 + 1
    #df_dummy.loc[:,('month_of_year')]=df_dummy.time.dt.month
    #df_dummy.loc[:,('quarter_of_year')]=df_dummy.time.dt.quarter
    df_dummy.loc[:,('day_of_month')] = df_dummy.time.dt.day
    
    df_dummy['date'] = pd.to_datetime(df_dummy.time.dt.strftime('%Y-%m-%d'))
    df_dummy = df_dummy.merge(holidays,on='date', how='left').drop(columns=['date'])
    df_dummy.loc[:, ('is_holiday')].fillna(0, inplace=True)

    dummy_columns = ['host', 'consumer_group', 'hour_of_day', 'day_of_week', 'day_of_month', 'week_of_month']
    df_dummy = pd.concat([df_dummy[['host', 'consumer_group']], pd.get_dummies(df_dummy.astype(str),
                             columns=dummy_columns,
                             prefix=dummy_columns,
                             drop_first=True)], axis=1, sort=False)
    
    return df_dummy

def lag_features(df):
    
    return df

if __name__ == "__main__":
    
    df_sysMetrics = process_files(file_to_df('input/GrpStat_OSGLOB*.dat'), drop_columns=['time'])
    df_grpMetrics = process_files(file_to_df('input/GrpStat_RGALL*.dat'), drop_columns=['t_beg'], rename_columns={"instance_number": "host", "usr_group": "consumer_group"})
    df_directives = process_files(file_to_df('input/GrpStat_Directives*.dat', time_list=['begin_time']), rename_columns={"instance_number": "host", "begin_time": "time"})

    df = df_grpMetrics.set_index(['host','snap_id','consumer_group'])         .join(df_directives.set_index(['host','snap_id','consumer_group']))        .reset_index()        .merge(df_sysMetrics,on=['host','snap_id'])
        
    df = prepare_features(df, group_excl=['other_groups','ods2exa_group'])
    df = sample_features(df, sample_period='H')
    df = scale_features(df)
    #df = dummy_features(df)
    df = lag_features(df)
    
    out_name = 'rm_features.csv'
    out_dir = 'clear_data'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    full_out_name = os.path.join(out_dir, out_name)
    
    df.to_csv(full_out_name, ';', index=False)

