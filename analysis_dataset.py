import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from utils import get_variants_percentage
import pandas as pd 
import numpy as np
import plotly.graph_objects as go
from fastDamerauLevenshtein import damerauLevenshtein as DL
from helpdesk_utils import helpdesk_utils
from bpic2012_utils import bpic2012_utils
from tqdm import tqdm
import os
from plotly.subplots import make_subplots

datasets = ['helpdesk', 'bpic2012']
try:
    os.remove('datasets.stat')
except:
    print('created new file stat')
#fig = make_subplots(rows=1, cols=2, shared_yaxes=True)
for dataset in datasets:
    fig = go.Figure()
    if dataset == 'helpdesk':
        log_csv = pd.read_csv('data/finale.csv' , sep=',')
        # Create the Event Log object as in the library pm4py
        log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
        log_csv = log_csv.sort_values('Complete Timestamp')
        # Mapping case name reflecting temporal order
        case_ids = log_csv.drop_duplicates('Case ID')
        case_ids['case:concept:name'] = np.arange(1, len(case_ids)+1)
        mapping = dict(zip(case_ids['Case ID'], case_ids['case:concept:name'].map(str)))
        log_csv['case:concept:name'] = log_csv['Case ID'].map(mapping)  
        log_csv.rename(columns={'Complete Timestamp': 'time:timestamp',
                                'Activity': 'concept:name'}, inplace=True)
        parameters = {
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case:'}
        event_log = log_converter.apply(log_csv, parameters=parameters,
                                        variant=log_converter.Variants.TO_EVENT_LOG)
        _, _, vocabulary, _, _ = helpdesk_utils(False)
        col = 1
    elif dataset == 'bpic2012':
        event_log = pm4py.read_xes('data/BPI_Challenge_2012.xes')
        for trace in event_log:
            trace.attributes['Case ID'] = trace.attributes['concept:name']
            for event in trace:
                event['concept:name'] = '{}_{}'.format(event['concept:name'], event['lifecycle:transition'])
                event['Case ID'] = 'Case {}'.format(trace.attributes['concept:name'])
        _, _, vocabulary, _, _, _ = bpic2012_utils(False)
        col = 2

    variants_perc, variants_count = get_variants_percentage(event_log)
    data_y = list()
    for variant in variants_perc:
        data_y.append(variant['count'])
    #fig.add_trace(go.Bar(y=data_y[:25], marker=dict(color='blue')), row=1, col=col)
    fig.add_trace(go.Bar(y=data_y[:25], marker=dict(color='blue')))
    for i in np.arange(500, 3500+500, 500):
        fig.add_hline(y=i, line_width=0.4, line_dash='dash', line_color='black')
    fig.update_layout(paper_bgcolor='rgba(255,255,255,1)', plot_bgcolor='rgba(255,255,255,1)', font=dict(size=18))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', range=[0, 3600])
    fig.write_image('saved_figures/datasets/{}_25_variants.svg'.format(dataset))
    fig.show()
#    fig.write_html('saved_figures/{}_25_variants.html'.format(dataset))
#    fig.write_image('saved_figures/{}_25_variants.svg'.format(dataset))

    variant_length = list()
    for trace in event_log:
        variant_length.append(len(trace))
    #breakpoint()
    variants_numerical = [list(map(lambda x: vocabulary.index(x), variant['variant'].split(','))) for variant in variants_perc] 
    variants_numerical_count = [variant['count'] for variant in variants_perc] 
    dl_distance = list()
    variant_num = 1
    for variant in tqdm(variants_numerical):
        for variant2 in variants_numerical[variant_num:]:
            dl_distance.append(DL(variant, variant2))
        variant_num += 1
    with open('datasets.stat', 'a') as file:
        file.write('--- {} ---\n'.format(dataset.capitalize()))
        file.write('mean variant length: {} +- {}\n'.format(
            np.round(np.mean(variant_length), 2), np.round(np.std(variant_length), 2)))
        file.write('mean dl: {} +- {}\n'.format(np.round(np.mean(dl_distance), 2), np.round(np.std(dl_distance), 2)))
        file.write('vocabulary length: {}\n'.format(len(vocabulary)))
        file.write('number of variants: {}\n'.format(len(variants_perc)))
        file.write('number of traces :{}\n'.format(len(event_log)))
        file.write('percentage of second variant wrt first: {}\n'.format(
            np.round(variants_numerical_count[1]/variants_numerical_count[0], 4)))
#fig.update_layout(showlegend=False)
#fig.write_image('saved_figures/dataset_25_variants.svg')
#fig.show()
