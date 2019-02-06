import colorlover as cl
import random
import pandas as pd
import numpy as np
import itertools

from plotly import graph_objs as go
from plotly import tools

from datetime import datetime, timedelta, date
from collections import defaultdict

def show_forecast(X, y, button_name_prefix, graph_name_prefix, chart_name, anomaly=None, timeinterval=None, date_range=None):
    ''' Visualization function
    '''

    colors = [color for color in cl.flipper()['seq']['9'].values()]
    data = defaultdict(list)

    for i, value in enumerate([(key, value) for key, value in y.items()]):
        button, dict_graphs = value
        fact_data = []
        if i == 0: 
            ButtonVisible = True
        else: 
            ButtonVisible = False
		
		# Аномальные значения
        if anomaly is None:
            anomaly_data = []
        else:
            anomaly_data = [go.Scatter(
                x = [anomaly[button][i], (anomaly[button][i] + timeinterval)[0]],
                y = [1,1],
                fill = 'tozeroy',
                fillcolor = 'rgba(190,127,188,0.5)',
                line = dict(width=0),
                mode = 'none',
                legendgroup = 'anomaly',
                name = 'anomaly',
                visible = ButtonVisible,
                showlegend = True if i == 0 else False
            ) for i in range(len(anomaly[button]))]

        # фактические значения
        for j, value in enumerate([(key, value) for key, value in dict_graphs.items()]):
            graph, list_values = value
            
            dash='longdash'
            
            if j%2==0:
                dash='solid'
            elif j%3==0:
                dash='dash'
            elif j%5==0:
                dash='dot'

            
            if (ButtonVisible == True) & (j != 0): 
                ButtonVisible = 'legendonly'
                
            colorpal = random.randint(0, len(colors) - 1)
            colorintensity = random.randint(2, 8)
            fact_data.append(go.Scatter(
                name=graph_name_prefix+str(graph),
                x = X[button][graph],
                y = y[button][graph],
                mode='lines',
                line=dict(color=colors[colorpal][colorintensity],
                          dash=dash,
                          width=2
                           ),
                visible=ButtonVisible
                )) 

        data[button]=list(filter(None.__ne__,[*fact_data, *anomaly_data]))
        
    
    updatemenus = list([
    dict(type="buttons",
         x = -0.07,
         buttons=list([
        dict(label=button_name_prefix + str(button),
          method = 'update',
          args = [
              {'visible':list(itertools.chain.from_iterable([([True]+
                                                              (len(y[key])-1)*['legendonly']+
                                                              (len(values)-len(y[key]))*[True]) 
                                                             
                                                             if key==button 
                                                             else len(values)*[False] 
                                                             for key, values in data.items()]
          )) },
             ])
        for i, button in enumerate([key for key in y.keys()]) 
         ])
        )
 ])

    layout = dict(title=chart_name, 
                  showlegend=True,
                  updatemenus=updatemenus,

                  xaxis=dict(
                      range = date_range,
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

    return dict(data=list(itertools.chain.from_iterable([value for key,value in data.items()])), layout=layout)


