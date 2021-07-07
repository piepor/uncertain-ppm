# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import numpy as np
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go
import pickle

#df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
colors = ['red', 'blue', 'grey', 'green', 'brown',
          'cadetblue', 'navy', 'cornflowerblue', 'darkslategrey', 'teal']
data_app_dir = 'app_data'
file_name = os.path.join(data_app_dir, 'df_app.pickle')
df = pd.read_pickle(file_name)
file_name = os.path.join(data_app_dir, 'df_pred.pickle')
df_pred = pd.read_pickle(file_name)
file_name = os.path.join(data_app_dir, 'vocabulary_app.pickle')
vocabulary = pd.read_pickle(file_name)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True

variable_indicators = [('Primo', 1), ('Secondo', 2)]
app.layout = html.Div([
    html.Div([
        dcc.Dropdown(id='variable-choice', 
                     options=[{"label": i, "value": j} for i,j in variable_indicators]),
    ], style={"width":"50%"}),
    html.Div([], 'example-graph') #, style={"height" : 800, "width" : "75%"})
])

@app.callback(
    Output({'type':'graph', 'index':MATCH}, 'figure'),
    Input({'type':'step-slider',  'index':MATCH}, 'value'))
def attention_box_plot(num_step):
    num_samples = len(df['num_sample'].unique())
    num_layers = len(df['layer'].unique())
    num_heads = len(df['head'].unique())
    activities_role_name = []
    for i in range(num_step):
        activities_role_name.extend(num_samples*[df['act_role'].unique()[i]])
    fig = make_subplots(rows=num_layers, cols=1)
    cont = 0
    for layer in df['layer'].unique():
        cont += 1
        if layer == 1:
            legend = True
        else:
            legend = False
        for head in df['head'].unique():
            data = []
            for k in df['act_role'].unique():
               data.extend(df.loc[(df['layer'] == layer) & \
                                  (df['num_step'] == num_step) & \
                                  (df['head'] == head) & \
                                  (df['act_role'] == k)]['att_weight'])
            fig.add_trace(
                go.Box(x=activities_role_name, y=data, name='head{}'.format(str(head)),
                       showlegend=legend, offsetgroup=head-1, marker_color=colors[head-1]),
                row=cont, col=1)    
            fig.update_layout(boxmode='group')
            fig.update_yaxes(range=[-0.1, 1.1])
    return fig

@app.callback(
    Output({'type':'statistics', 'index':MATCH}, 'figure'),
    Input({'type':'step-slider',  'index':MATCH}, 'value'))
def statistic_bar_plot(num_step):
    num_samples = len(df['num_sample'].unique())
    activities_role_name = []
    actual_value = df_pred.loc[(df_pred['num_step']==num_step)]['actual_value'].values[0]
    colors_pred = []
    for i in vocabulary['activity-role-index'].sort_values():
        act_rol_idx = vocabulary[vocabulary['activity-role-index']==i]
        activities_role_name.extend(['{} - role {}'.format(act_rol_idx['task'].values[0], act_rol_idx['role'].values[0])])
        if i == int(actual_value):
            colors_pred.extend(['red'])
        else:
            colors_pred.extend(['blue'])
    fig = make_subplots(rows=2, cols=1)
    cont = 0
    prediction_entropy = []
    for row in range(2):
        cont += 1
        if row == 0:
            # add dummy bar chart just to set the right legend
            fig.add_trace(go.Bar(x=['<PAD> - role 0'], y=[0], name='Actual Value', marker_color='red'), row=cont, col=1)
            fig.add_trace(go.Bar(x=['<PAD> - role 0'], y=[0], name='Predicted Distribution', marker_color='blue'), row=cont, col=1)
            for k in df_pred['num_sample'].unique():
                data = df_pred.loc[(df_pred['num_step']==num_step) & (df_pred['num_sample']==k)]['percentage'].values
                prediction_entropy.append(- np.sum(data*np.log(data)))
                fig.add_trace(
                    go.Bar(x=activities_role_name, y=data, marker_color=colors_pred, opacity=0.1, showlegend=False),
                    row=cont, col=1)    
                fig.update_layout(barmode='overlay')
            fig.update_yaxes(range=[-0.1, 1.1], row=cont, col=1)
        else:
                fig.add_trace(
                    go.Box(y=prediction_entropy, showlegend=False),
                    row=cont, col=1)    
                fig.update_yaxes(range=[-0.1, 2], row=cont, col=1)

    return fig
#@app.callback(
#    Output({'type':'graph', 'index':MATCH}, 'figure'),
#    Input({'type':'year-slider',  'index':MATCH}, 'value'))
#def update_figure(selected_year):
#    filtered_df = df[df.year == selected_year]
#
#    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",
#                     size="pop", color="continent", hover_name="country",
#                     log_x=True, size_max=55)
#
#    fig.update_layout(transition_duration=500)
#
#    return fig

@app.callback(
    Output('example-graph', 'children'),
    [Input('variable-choice', 'value')],
    [State('example-graph', 'children')]
)
def select_graph(select_graph, children):
    if select_graph == 1:
        new_element = html.Div([
            dcc.Graph(id={'type': 'graph', 'index':len(children)+1}, figure={"layout":{"height": 1500, "width":2100}}),
            dcc.Slider(
                id={'type': 'step-slider', 'index':len(children)+1},
                min=df['num_step'].min(),
                max=df['num_step'].max(),
                value=df['num_step'].min(),
                marks={int(step): "step {}".format(int(step)) for step in df['num_step'].unique().tolist()})
        ],
        style={"display":"inline-block"})
        children.append(new_element)
        new_element = html.Div([
            dcc.Graph(id={'type': 'statistics', 'index':len(children)}, figure={"layout":{"height": 1500, "width":1000}}),
        ], style={"display":"inline-block", "align":"right", "vertical-align": "top"})
        children.append(new_element)
                                                                                       
    else:
        children = []
    return children
if __name__ == '__main__':
    app.run_server(debug=True)
