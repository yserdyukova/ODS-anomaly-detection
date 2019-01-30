import pandas as pd
import numpy as np
import itertools
import colorlover as cl
import random
import sys

from datetime import datetime, timedelta, date
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from plotly.offline import init_notebook_mode, plot, iplot
from plotly import graph_objs as go

from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping

import dash
import dash_core_components as dcc
import dash_html_components as html


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean(axis=1))

def mse(predictions, targets):
    return((predictions - targets) ** 2).mean(axis=1)

def mae(predictions, targets):
    return (abs(predictions - targets)).mean(axis=1)



def noise_repeat(X, noise=0.2, repeat=5):
    
    X_real = np.repeat(X, repeat, axis=0)
    noise_array = noise * np.random.normal(loc=0.0, scale=1.0, size=[X.shape[i] * repeat if i == 0 else X.shape[i] if i == 1 else 1 for i in range(len(X.shape))])
    X_noise = np.clip(X_real + noise_array, 0, 1)
    
    return X, X_real, X_noise


def autoencoder_fit(X_predict, X_real, X_noise, model, verbose=0):
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    earlystopper = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_noise, 
                    X_real,
                    epochs=1000,
                    batch_size=128,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=verbose,
                    callbacks=[earlystopper])
    
    return model.predict(X_predict)


def noise_autoencoder(dim):
    
    X_input = Input(shape=(dim,))
    encoded = Dense(128, activation='relu')(X_input)
    encoded = Dense(64, activation='linear', activity_regularizer=regularizers.l1(10e-5))(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(X_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    return autoencoder




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



def dash_create(figure):

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for Python.
        '''),

        dcc.Graph(
            id='example-graph',
            figure = figure
        )
    ])
	
    return app


if __name__ == "__main__":
    
    df = pd.read_csv('../features/clear_data/rm_features.csv', ';', infer_datetime_format=True, parse_dates=['time'])

    # Noise AutoEncoder
    X = df.set_index(['time', 'host', 'consumer_group'])
    X_predict, X_real, X_noise = noise_repeat(X.values)
    noise_model = noise_autoencoder(X_predict.shape[1])
    autoencoder_predict = autoencoder_fit(X_predict, X_real, X_noise, noise_model, verbose=0)
    autoencoder_error = pd.concat([X.reset_index()[['time','host','consumer_group']], pd.DataFrame(mse(autoencoder_predict, X.values), columns=['error'])], axis=1, sort=False)
   
    autoencoder_error['is_anomaly'] = autoencoder_error.groupby(['host','consumer_group'])['error'].transform(lambda x: np.abs(x - x.mean()) > 3 * x.std())

    anomaly_predict = autoencoder_error[autoencoder_error.is_anomaly == True]
    
    # Visualization
    #for consumer_group in anomaly_predict.consumer_group.unique():
        #fig_reqs = show_forecast(X.query('consumer_group == @consumer_group').reset_index(), X.columns, consumer_group, anomaly_predict[anomaly_predict.consumer_group == consumer_group])
        #iplot(fig_reqs)

        #plot(fig_reqs, filename='../results/report_{0}.html'.format(consumer_group))

    consumer_group = 'szb_consumer_group'
    fig_reqs = show_forecast(X.query('consumer_group == @consumer_group').reset_index(), X.columns, consumer_group, anomaly_predict[anomaly_predict.consumer_group == consumer_group])
    
    dash_create(fig_reqs).run_server(debug=True)
	

