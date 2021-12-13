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
    fig.add_trace(go.Scatter(x=np.cumsum(rel_bin_plot)-rel_bin_plot, y=perc_data,
        name='perc data ens', line=dict(color='black'), mode='lines+markers'))
    fig.add_trace(go.Scatter(x=np.cumsum(rel_bin_plot)-rel_bin_plot, y=perc_data_one_model,
        name='perc data one model', line=dict(color='black', dash='dot'), mode='lines+markers',
        marker=dict(symbol='cross')))
    fig.update_layout(title_text='Reliability Diagram - {}'.format(title_text))
    fig.update_xaxes(title_text='Confidence')
    fig.update_yaxes(title_text='Accuracy')
    fig.show(renderer='chromium')

def mean_accuracy_plot(unc, acc, unc_type, title_text):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_t_array, y=acc_array, mode='markers',
                             name='{} uncertainty'.format(unc_type)))
    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text='{} uncertainty'.format(unc_type))
    fig.update_yaxes(title_text='Mean sequence accuracy')
    fig.show(renderer='chromium')

def proportions_plot(u_t_plot, perc_rigth_plot, perc_wrong_plot, title_text):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.cumsum(u_t_plot)-u_t_plot, y=perc_right_plot, 
                         width=u_t_plot, offset=0, name='right predictions'))
    fig.add_trace(go.Bar(x=np.cumsum(u_t_plot)-u_t_plot, y=perc_wrong_plot,
                         width=u_t_plot, offset=0, name='wrong predictions'))
    fig.update_layout(title_text=title_text)
    fig.show(renderer='chromium')
    return fig

def accuracy_uncertainty_plot(u_t_plot, acc_plot, perc_data, title_text):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.cumsum(u_t_plot)-u_t_plot, y=acc_plot, 
                         width=u_t_plot, offset=0, name='accuracy'))
    fig.add_trace(go.Bar(x=np.cumsum(u_t_plot)-u_t_plot, y=perc_data, 
                         width=u_t_plot, offset=0, name='data percentage'))
    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text='total uncertainty')
    fig.update_yaxes(title_text='accuracy')
    fig.show(renderer='chromium')
    return fig

def event_probability_plot(total_label, u_t_array_single_right, u_t_array_single_wrong,
                           target_prob_array, acc_array_single, multi_entropy_arrays, prob_array,
                           vocabulary_act, bin_entropy_array, u_a_array_single_right,
                           u_a_array_single_wrong, u_e_array_single_right, u_e_array_single_wrong,
                           max_prob_event_array, title_text, entropy_other, target_label_array, prob):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_t_array_single_right, y=target_prob_array[acc_array_single==1],
        mode='markers', text=total_label, name='right prediction'))
    fig.add_trace(go.Scatter(x=u_t_array_single_wrong, y=target_prob_array[acc_array_single==0],
        mode='markers', text=total_label, name='wrong prediction'))
    fig.add_trace(go.Scatter(x=bin_entropy_array, y=prob_array,
                  name='Binary entropy'))
    for i in range(len(multi_entropy_arrays)):
        number_of_classes = len(vocabulary_act)
        fig.add_trace(go.Scatter(x=multi_entropy_arrays[i], y=prob_array,
                      name='Max entropy {} class'.format(number_of_classes-i)))
    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text='Total uncertainty')
    fig.update_yaxes(title_text='Assigned probability')
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_a_array_single_right, y=target_prob_array[acc_array_single==1],
        mode='markers', text=target_label_array.tolist(), name='right prediction'))
    fig.add_trace(go.Scatter(x=u_a_array_single_wrong, y=target_prob_array[acc_array_single==0],
        mode='markers', text=target_label_array.tolist(), name='wrong prediction'))
    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text='Aleatoric uncertainty')
    fig.update_yaxes(title_text='Assigned probability')
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_e_array_single_right, y=target_prob_array[acc_array_single==1],
        mode='markers', text=target_label_array.tolist(), name='right prediction'))
    fig.add_trace(go.Scatter(x=u_e_array_single_wrong, y=target_prob_array[acc_array_single==0],
        mode='markers', text=target_label_array.tolist(), name='wrong prediction'))
    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text='Epistemic uncertainty')
    fig.update_yaxes(title_text='Assigned probability')
    fig.show(renderer='chromium')

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

def event_correctness_plot(u_t_array_single, u_a_array_single, u_e_array_single,
                           acc_array_single, target_label_array, target_prob_array, title_text):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_t_array_single[acc_array_single>-1], y=acc_array_single[acc_array_single>-1],
        marker=dict(color=target_prob_array, colorbar=dict(title='Assigned probability'), colorscale='Viridis'),
        mode='markers', text=target_label_array.tolist()))
    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text='Total uncertainty')
    fig.update_yaxes(title_text='Point prediction correctness')
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_a_array_single[acc_array_single>-1], y=acc_array_single[acc_array_single>-1],
        marker=dict(color=target_prob_array, colorbar=dict(title='Assigned probability'), colorscale='Viridis'),
        mode='markers', text=target_label_array.tolist()))
    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text='Aleatoric uncertainty')
    fig.update_yaxes(title_text='Point prediction correctness')
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=u_e_array_single[acc_array_single>-1], y=acc_array_single[acc_array_single>-1],
        marker=dict(color=target_prob_array, colorbar=dict(title='Assigned probability'), colorscale='Viridis'),
        mode='markers', text=target_label_array.tolist()))
    fig.update_layout(title_text=title_text)
    fig.update_xaxes(title_text='Epistemic uncertainty')
    fig.update_yaxes(title_text='Point prediction correctness')
    fig.show(renderer='chromium')

def distributions_plot(u_t_array_single, u_a_array_single, u_e_array_single, title_text):
    group_labels = ['Total uncertainty', 'Aleatoric uncertainty', 'Epistemic uncertainty']
    hist_data = [u_t_array_single, u_a_array_single, u_e_array_single]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
    fig.update_layout(title_text=title_text)
    fig.show(renderer='chromium')

def distributions_plot_right_wrong(u_t_array_single_right, u_t_array_single_wrong,
                                   u_a_array_single_right, u_a_array_single_wrong,
                                   u_e_array_single_right, u_e_array_single_wrong, title_text):
    group_labels = ['Right predictions', 'Wrong predictions']
    hist_data = [u_t_array_single_right, u_t_array_single_wrong]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.05)
    fig.update_layout(title_text="{} - Total uncertainty".format(title_text))
    fig.show(renderer='chromium')

    hist_data = [u_a_array_single_right, u_a_array_single_wrong]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.05)
    fig.update_layout(title_text="{} - Aleatoric uncertainty".format(title_text))
    fig.show(renderer='chromium')

    hist_data = [u_e_array_single_right, u_e_array_single_wrong]
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.05)
    fig.update_layout(title_text="{} - Epistemic uncertainty".format(title_text))
    fig.show(renderer='chromium')

def box_plot_func(u_t_array_single_right, u_t_array_single_wrong, u_t_array_single,
                  u_a_array_single_right, u_a_array_single_wrong, u_a_array_single,
                  u_e_array_single_right, u_e_array_single_wrong, u_e_array_single,
                  title_text):
    fig = go.Figure()
    fig.add_trace(go.Box(y=u_t_array_single_right, name='Right predictions'))
    fig.add_trace(go.Box(y=u_t_array_single_wrong, name='Wrong predictions'))
    fig.update_layout(title_text="{} - Total uncertainty".format(title_text))
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_a_array_single_right, name='Right predictions'))
    fig.add_trace(go.Box(y=u_a_array_single_wrong, name='Wrong predictions'))
    fig.update_layout(title_text="{} - Aleatoric uncertainty".format(title_text))
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_e_array_single_right, name='Right predictions'))
    fig.add_trace(go.Box(y=u_e_array_single_wrong, name='Wrong predictions'))
    fig.update_layout(title_text="{} - Epistemic uncertainty".format(title_text))
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_t_array_single, name='Total uncertainty'))
    fig.update_layout(title_text=title_text)
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_a_array_single, name='Aleatoric uncertainty'))
    fig.update_layout(title_text=title_text)
    fig.show(renderer='chromium')

    fig = go.Figure()
    fig.add_trace(go.Box(y=u_e_array_single, name='Epistemic uncertainty'))
    fig.update_layout(title_text=title_text)
    fig.show(renderer='chromium')

def sequences_plot(prob_unc_mask, acc_single, check_cond_wrong, random_idx_plot, input_data, u_t,
                   u_a, batch_size, plot_threshold, target_data, mask, plot_wrong_preds,
                   vocabulary_plot, out_prob, out_prob_tot_distr, model_type, 
                   batch_idx, title_text, count_wrong, count_seq, num_wrong_preds):
    if not np.ma.masked_equal(prob_unc_mask, True).mask.any():
        prob_unc_mask = np.zeros_like(acc_single)
    for num_row in range(acc_single.shape[0]):
        check_cond_wrong_row = np.logical_and(acc_single[num_row, :]==0.0, mask[num_row, :]).any() and check_cond_wrong
        if plot_wrong_preds:
            check_cond_random_idx = (batch_idx * batch_size + num_row in random_idx_plot) 
        else:
            check_cond_random_idx = False
        if check_cond_wrong_row or check_cond_random_idx or (prob_unc_mask[num_row, 0] and plot_threshold):
            act = ''
            for j in range(target_data.shape[1]):
                act += '{} - '.format(input_data[0][num_row, j].numpy().decode("utf-8"))
                check_cond_wrong_single_pred = check_cond_wrong_row and acc_single[num_row, j] == 0.0 and count_wrong<num_wrong_preds
                if (check_cond_wrong_single_pred or check_cond_random_idx or (plot_threshold and prob_unc_mask[num_row, 0])) and not target_data[num_row, j].numpy() == 0:
                    target_numpy = np.zeros(len(vocabulary_plot))
                    target_numpy[target_data[num_row, j].numpy()] = 1
                    fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2])
                    fig.add_trace(go.Bar(x=vocabulary_plot, y=target_numpy,
                                         marker=dict(opacity=0.4), name='actual event'),
                                  row=1, col=1)
                    fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob[num_row, j].numpy(),
                                         marker=dict(opacity=0.4), name='single model'),
                                  row=1, col=1)
                    fig.add_trace(go.Bar(x=vocabulary_plot, y=out_prob_tot_distr[num_row, j].numpy(),
                                         marker=dict(opacity=0.4), name='{} models'.format(model_type)),
                                  row=1, col=1)
                    fig.add_trace(go.Bar(x=['Uncertainty'], y=[u_t[num_row, j]], name='Epistemic', offsetgroup=0),
                                  row=1, col=2)
                    fig.add_trace(go.Bar(x=['Uncertainty'], y=[u_a[num_row, j]], name='Aleatoric', offsetgroup=0),
                                  row=1, col=2)
                    fig.layout['yaxis2'].update(range=[0, 1.2*np.max(u_t)])
                    fig.update_layout(barmode='overlay', title_text=title_text)
                    #fig.update_traces(opacity=0.6)
                    fig.show(renderer='chromium')
                    #breakpoint()
                    count_wrong += 1
            count_seq += 1