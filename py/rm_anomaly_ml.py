#!/usr/bin/env python3

import pandas as pd
import numpy as np
import itertools
import colorlover as cl
import random
import plotly.graph_objs as go
import sys

from datetime import datetime, timedelta, date
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from plotly.offline import init_notebook_mode, plot, iplot

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras import backend as K

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
init_notebook_mode(connected=True)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=1))

def mse(predictions, targets):
    return((predictions - targets) ** 2).mean(axis=1)

def mae(predictions, targets):
    return (abs(predictions - targets)).mean(axis=1)

def noise_autoencoder(X, noise=0, repeat=3):
    
    dim = X.shape[1]    
    
    X_real = np.repeat(X, repeat, axis=0)
    noise_array = noise * np.random.normal(loc=0.0, scale=1.0, size=[X.shape[0] * 3, X.shape[1]])
    X_noise = np.clip(X_real + noise_array, 0, 1)
    
    X_input = Input(shape=(dim,))
    encoded = Dense(128, activation='relu')(X_input)
    encoded = Dense(64, activation='linear', activity_regularizer=regularizers.l1(10e-5))(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(X_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    earlystopper = EarlyStopping(monitor='val_loss', patience=5)
    autoencoder.fit(X_noise, 
                    X_real,
                    epochs=1000,
                    batch_size=128,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[earlystopper])

    return autoencoder.predict(X)


def show_forecast(X, metrics, consumer_group, anomaly=None):
    ''' Visualization function
    '''
    
    # бага с legend doubleclick

    colors=[color for color in cl.flipper()['seq']['9'].values()]
    data_host=defaultdict(list)

    hosts=X.host.unique()
    for i,host in enumerate(hosts):
        fact_data=[]
        if i==0: 
            ButtonVisible=True
        else: 
            ButtonVisible=False

        anomaly_data = [go.Scatter(
            x=[anomaly[anomaly.host==host].iloc[i].time,anomaly[anomaly.host==host].iloc[i].time+timedelta(hours=1)],
            y=[1,1],
            fill='tozeroy',
            fillcolor='rgba(190,127,188,0.5)',
            line=dict(width=0),
            mode= 'none',
            showlegend=False,
            visible=ButtonVisible
        ) for i in range(len(anomaly[anomaly.host==host]))]
        
           
        # фактические значения
        for j,metric in enumerate(metrics):
            
            dash='longdash'
            
            if j%2==0:
                dash='solid'
            elif j%3==0:
                dash='dash'
            elif j%5==0:
                dash='dot'

            
            if (ButtonVisible==True) & (j!=0): 
                ButtonVisible='legendonly'
                
            colorpal=random.randint(0,len(colors)-1)
            fact_data.append(go.Scatter(
                name=str(metric),
                #legendgroup=str(metric),     
                #showlegend= False,
                x=X[X.host==host].time,
                y=X[X.host==host][metric].values,
                mode='lines',
                line=dict(color=colors[colorpal][i+3],
                          dash=dash,
                          width=2
                           ),
                visible=ButtonVisible
                )) 

        data_host[host]=list(filter(None.__ne__,[*fact_data,*anomaly_data]))

    updatemenus = list([
    dict(type="buttons",
         x = -0.07,
         buttons=list([
        dict(label='Host '+str(hostname),
          method = 'update',
          args = [
              {'visible':list(itertools.chain.from_iterable([([True]+(len(metrics)-1)*['legendonly']+(len(values)-len(metrics))*[True]) if host==hostname else len(values)*[False] for host,values in data_host.items()]
          )) },
             ])
        for i,hostname in enumerate(hosts) 
         ])
        )
 ])


    layout = dict(title=consumer_group, 
                  showlegend=True,
                  updatemenus=updatemenus,

                  xaxis=dict(
                      range=['2018-11-12','2018-11-14'],
                      rangeselector=dict(
                          buttons=list([
                              dict(count=1,
                                   label='1d',
                                   step='day',
                                   stepmode='backward'),
                              dict(count=7,
                                   label='1w',
                                   step='day',
                                   stepmode='backward'),
                              dict(count=1,
                                   label='1m',
                                   step='month',
                                   stepmode='backward'),
                              dict(step='all',
                                   stepmode='backward')
                          ]),
                      ),
                      rangeslider=dict(
                          visible = True
                      ),
                      type='date'
                  ),
                  yaxis=dict(
                      ticks='outside',
                      zeroline=False
                  ),
                 )
    return dict(data=list(itertools.chain.from_iterable([value for key,value in data_host.items()])), layout=layout)

if __name__ == "__main__":
    
    df = pd.read_csv('../features/clear_data/rm_features.csv', ';', infer_datetime_format=True, parse_dates=['time'])
    X = df.set_index(['time', 'host', 'consumer_group'])

    # AutoEncoder
    autoencoder_predict = noise_autoencoder(X.values, 0.2)
    autoencoder_error = pd.concat([X.reset_index()[['time','host','consumer_group']], pd.DataFrame(mse(autoencoder_predict, X.values), columns=['error'])], axis=1, sort=False)
    outliers_fraction = 0.015
    autoencoder_error['quantile_error'] = autoencoder_error.groupby(['host','consumer_group'])['error'].transform(lambda x: x.quantile(1 - outliers_fraction))
    
    anomaly_predict = autoencoder_error.query('error > quantile_error')
    plt.plot(autoencoder_error[['time']], autoencoder_error[['error']], '*')
    plt.xticks(rotation='vertical')  
    
    # Visualization
    for consumer_group in anomaly_predict.consumer_group.unique():
        fig_reqs = show_forecast(X.query('consumer_group == @consumer_group').reset_index(), X.columns, consumer_group, anomaly_predict[anomaly_predict.consumer_group == consumer_group])
        #iplot(fig_reqs)

        plot(fig_reqs, filename='../results/report_{0}.html'.format(consumer_group))

