import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from plotly.subplots import make_subplots

def reliability_diagram_plot(rel_dict, rel_dict_one_model, rel_bins, title_text, ece=None, ece_one_model=None):
    rel_bin_plot = []
    rel_acc_plot = []
    rel_acc_plot_one_model = []
    perc_data = []
    perc_data_one_model = []
    num_total_valid_data = 0
    num_total_valid_data_one_model = 0
    for key in rel_dict.keys(): 
        num_total_valid_data += rel_dict[key][:, 2].sum() 
        num_total_valid_data_one_model += rel_dict_one_model[key][:, 2].sum() 
    for key in rel_dict.keys():
        rel_bin_plot.append(rel_bins)
        rel_acc_plot.append(np.mean(rel_dict[key][:, 0]))
        rel_acc_plot_one_model.append(np.mean(rel_dict_one_model[key][:, 0]))
        perc_data.append(rel_dict[key][:, 2].sum() / num_total_valid_data)
        perc_data_one_model.append(rel_dict_one_model[key][:, 2].sum() / num_total_valid_data_one_model)
    #breakpoint()
    name_ens = 'ensemble'
    name_one_mod = 'one model'
    if ece is not None:
        name_ens = '{} - ECE {}'.format(name_ens, np.round(ece, 4))
    if ece_one_model is not None:
        name_one_mod = '{} - ECE {}'.format(name_one_mod, np.round(ece_one_model, 4))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.cumsum(rel_bin_plot)-rel_bin_plot, y=rel_acc_plot, 
                         width=rel_bin_plot, offset=0, name=name_ens, opacity=0.7))
    fig.add_trace(go.Bar(x=np.cumsum(rel_bin_plot)-rel_bin_plot, y=rel_acc_plot_one_model, 
                         width=rel_bin_plot, offset=0, name=name_one_mod, opacity=0.7))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color='black', dash='dash')))
    #fig.add_trace(go.Scatter(x=np.cumsum(rel_bin_plot)-rel_bin_plot, y=perc_data,
    #    name='perc data ens', line=dict(color='black'), mode='lines+markers'))
    #fig.add_trace(go.Scatter(x=np.cumsum(rel_bin_plot)-rel_bin_plot, y=perc_data_one_model,
    #    name='perc data one model', line=dict(color='black', dash='dot'), mode='lines+markers',
    #    marker=dict(symbol='cross')))
    fig.add_hline(y=1, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.8, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.6, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.4, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.2, line_width=0.4, line_dash='dash', line_color='black')
    fig.update_layout(title_text='Reliability Diagram - {}'.format(title_text), paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)', font=dict(size=18))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',
            title_text='Confidence', range=[0, 1.1])
    if 'Helpdesk' in title_text:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', rangemode='tozero')
    else:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                title_text='Accuracy', range=[0, 1.1])
    fig.show(renderer='chromium')
    fig.write_image('saved_figures/models/reliability-diagram-{}.svg'.format(''.join(title_text.split())))
    fig.write_html('saved_figures/models/reliability-diagram-{}.html'.format(''.join(title_text.split())))

def accuracy_uncertainty_plot(u_t_plot, acc_plot, perc_data, title_text):
    colors = list()
    for i in range(len(acc_plot)):
        colors.append('rgba(158,202,225,{})'.format(np.round(perc_data[i], 4)+0.1))
    #breakpoint()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.cumsum(u_t_plot)-u_t_plot, y=acc_plot, 
                         width=u_t_plot, offset=0, name='accuracy', marker_color=colors))
    fig.add_hline(y=1, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.8, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.6, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.4, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.2, line_width=0.4, line_dash='dash', line_color='black')
    fig.update_layout(title_text=title_text, paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)', font=dict(size=18))
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',
            title_text='Total Uncertainty', range=[0, np.cumsum(u_t_plot)[-1]+0.1])
    if 'Helpdesk' in title_text:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', rangemode='tozero')
    else:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                title_text='Accuracy', range=[0,1.1])
    #fig.update_traces(marker_line_color='rgb(8,48,107)')
    fig.update_traces(marker_line_color='rgb(0,0,128)', marker_line_width=2)
    fig.show(renderer='chromium')
    fig.write_image('saved_figures/models/accuracy-uncertainty-{}.svg'.format(''.join(title_text.split())))
    fig.write_html('saved_figures/models/accuracy-uncertainty-{}.html'.format(''.join(title_text.split())))
    return fig

def event_probability_plot(total_label, u_t_array_single_right, u_t_array_single_wrong,
                           target_prob_array, acc_array_single, multi_entropy_arrays, prob_array,
                           vocabulary_act, bin_entropy_array, u_a_array_single_right,
                           u_a_array_single_wrong, u_e_array_single_right, u_e_array_single_wrong,
                           max_prob_event_array, title_text, entropy_other, target_label_array, prob):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_t_array_single_right, y=target_prob_array[acc_array_single==1],
        mode='markers', text=total_label, name='right prediction', marker_color='blue'))
    fig.add_trace(go.Scatter(x=u_t_array_single_wrong, y=target_prob_array[acc_array_single==0],
        mode='markers', text=total_label, name='wrong prediction', marker_color='red'))
    for i in range(len(multi_entropy_arrays)):
        number_of_classes = len(vocabulary_act)
        width = 1
        if i == 0 or i == len(multi_entropy_arrays)-1:
            width = 3
        fig.add_trace(go.Scatter(x=multi_entropy_arrays[i], y=prob_array,
                      name='Max entropy {} class'.format(number_of_classes-i), line=dict(color='black', width=width, dash='dot')))
    fig.add_vline(x=0.4, line_width=2, line_dash='dash', line_color='black')
    fig.add_hline(y=1, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.8, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.6, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.4, line_width=0.4, line_dash='dash', line_color='black')
    fig.add_hline(y=0.2, line_width=0.4, line_dash='dash', line_color='black')
    fig.update_layout(title_text=title_text, paper_bgcolor='rgba(255,255,255,1)',
            plot_bgcolor='rgba(255,255,255,1)', font=dict(size=18))
#    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',
#            title_text='Total uncertainty', rangemode='tozero', showgrid=True, gridwidth=1, gridcolor='black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black',
            title_text='Total uncertainty', rangemode='tozero')
    if 'Helpdesk' in title_text:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', rangemode='tozero')
    else:
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black',
                title_text='Assigned probability', rangemode='tozero')
    fig.add_annotation(x=0.57, y=0.5, text='Binomial<br>Entropy', showarrow=False)
    if number_of_classes == 16:
        fig.add_annotation(x=2.8, y=0.25, text='{}-classes Entropy'.format(number_of_classes), showarrow=False, yshift=10)
    elif number_of_classes ==38:
        fig.add_annotation(x=3.5, y=0.25, text='{}-classes Entropy'.format(number_of_classes), showarrow=False, yshift=10)
    fig.add_annotation(x=0.35, y=0.3, text='Threshold', showarrow=False, yshift=10, textangle=-90)
    fig.show(renderer='chromium')
    fig.write_image('saved_figures/models/event-probability-total-{}.svg'.format(''.join(title_text.split())))
    fig.write_html('saved_figures/models/event-probability-total-{}.html'.format(''.join(title_text.split())))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_t_array_single_right, y=max_prob_event_array[acc_array_single==1],
        mode='markers', text=total_label, name='right prediction'))
    fig.add_trace(go.Scatter(x=u_t_array_single_wrong, y=max_prob_event_array[acc_array_single==0],
        mode='markers', text=total_label, name='wrong prediction'))
    fig.add_trace(go.Scatter(x=bin_entropy_array, y=prob_array,
                  name='Binary entropy'))
    for i in range(len(multi_entropy_arrays)):
        fig.add_trace(go.Scatter(x=multi_entropy_arrays[i], y=prob_array,
                      name='Max entropy {} class'.format(i)))
    fig.add_trace(go.Scatter(x=entropy_other, y=prob,
                  name='Max entropy'))
    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text='Total uncertainty')
    fig.update_yaxes(title_text='Max probability predicted')
    fig.show(renderer='chromium')
    fig.write_html('saved_figures/models/event-probability-max-total-{}.html'.format(''.join(title_text.split())))

def plot_case(acc_single, input_data, u_t, target_data, vocabulary_plot,
        out_prob, out_prob_tot_distr, model_type, title_text, target_data_case, case_chosen):
    for num_row in range(acc_single.shape[0]):
        if target_data_case[num_row, 0] == case_chosen:
            act = ''
            for j in range(target_data.shape[1]):
                act += '{} - '.format(input_data[0][num_row, j].numpy().decode("utf-8"))
                if not target_data[num_row, j].numpy() == 0:
                    target_numpy = np.zeros(len(vocabulary_plot))
                    target_numpy[target_data[num_row, j].numpy()] = 1
                    fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2])
                    fig.add_trace(go.Bar(x=vocabulary_plot, y=target_numpy,
                                         marker=dict(color='white', line_color='black', line_width=2), name='actual event'),
                                  row=1, col=1)
                    fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob[num_row, j].numpy(),
                                         marker=dict(opacity=0.4, color='red'), name='single model'),
                                  row=1, col=1)
                    fig.add_hline(y=1, line_width=0.4, line_dash='dash', line_color='black')
                    fig.add_hline(y=0.8, line_width=0.4, line_dash='dash', line_color='black')
                    fig.add_hline(y=0.6, line_width=0.4, line_dash='dash', line_color='black')
                    fig.add_hline(y=0.4, line_width=0.4, line_dash='dash', line_color='black')
                    fig.add_hline(y=0.2, line_width=0.4, line_dash='dash', line_color='black')
                    fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob_tot_distr[num_row, j].numpy(),
                                         marker=dict(opacity=0.4, color='blue'), name='{} models'.format(model_type)),
                                  row=1, col=1)
                    fig.add_trace(go.Bar(x=['Total Uncertainty'], y=[u_t[num_row, j]], name='Total Uncertainty'),
                                  row=1, col=2)
                    for i in np.arange(0.5, 1.2*np.max(u_t)+0.5, 0.5):
                        fig.add_hline(y=i, line_width=0.4, line_dash='dash', line_color='black',
                                row=1, col=2)
                    fig.layout['yaxis2'].update(range=[0, 1.2*np.max(u_t)], title_text='[nats]')
                    fig.layout['yaxis1'].update(title_text='Probability')
                    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
                    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
                    fig.update_layout(barmode='overlay', title_text=title_text, showlegend=False,
                            paper_bgcolor='rgba(255,255,255,1)', plot_bgcolor='rgba(255,255,255,1)')
                    #fig.update_traces(opacity=0.6)
                    fig.show(renderer='chromium')
                    fig.write_html('saved_figures/models/case{}-{}-step{}.html'.format(case_chosen, ''.join(title_text.split()), j))
                    fig.write_image('saved_figures/models/case{}-{}-step{}.svg'.format(case_chosen, ''.join(title_text.split()), j))
                    #breakpoint()
