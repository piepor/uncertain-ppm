import pm4py

ds_helpdesk = pm4py.read_xes('data/Helpdesk.xes')

var_ids = set()

for trace in ds_helpdesk:
    var_ids.add(trace.attributes['variant-index'])

min_num_diff = 0
min_num = len(ds_helpdesk)
for var_id in var_ids:
    filt_ds = pm4py.filter_trace_attribute_values(ds_helpdesk, 'variant-index', {var_id})
    diff_traces = []
    for trace in filt_ds:
        act = []
        for event in trace:
            act.append(event['concept:name'])
        if not act in diff_traces:
            diff_traces.append(act)
        if len(diff_traces) > min_num_diff:
            min_num_diff = len(diff_traces)
            var_idx_min_num_diff = var_id
    num_var = len(filt_ds) 
    if num_var < min_num:
        min_num = num_var
        var_idx_min_num = var_id
