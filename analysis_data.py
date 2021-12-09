import pm4py
from pm4py.algo.organizational_mining.roles import algorithm as roles_discovery
from pm4py.algo.organizational_mining.sna import algorithm as sna
from pm4py.visualization.sna import visualizer as sna_visualizer

data_path = './data/BPI_Challenge_2012.xes'

log = pm4py.read_xes(data_path)

resources = set()
no_resources = set()
unique_activities = set()
unique_activities_resource = set()
count_no_resource = 0
count_w = 0
activities_stats = dict()
resources_stats = dict()

for trace in log:
    for event in trace:
        activity =(event['concept:name'], event['lifecycle:transition'])
        if 'org:resource' in event.keys():
            if 'W_' in event['concept:name']:
                count_w += 1
            if not event['org:resource'] in resources:
                resources.add(event['org:resource'])
            activity_resource = (event['org:resource'], event['lifecycle:transition'], event['concept:name'])
            if not activity_resource in unique_activities_resource:
                unique_activities_resource.add(activity_resource)
            key = '{}-{}'.format(activity[0], activity[1])
            if key in activities_stats.keys():
                if event['org:resource'] in activities_stats[key].keys():
                    activities_stats[key][event['org:resource']] += 1
                else:
                    activities_stats[key][event['org:resource']] = 1
            else:
                activities_stats[key] = {event['org:resource']:1}
            if event['org:resource'] in resources_stats.keys():
                if key in resources_stats[event['org:resource']].keys():
                    resources_stats[event['org:resource']][key] += 1
                else:
                    resources_stats[event['org:resource']][key] = 1
            else:
                resources_stats[event['org:resource']] = {key:1}
        else:
            event['org:resource'] = 'None'
            activity_resource = ('None', event['lifecycle:transition'], event['concept:name'])
            if not activity in no_resources:
                no_resources.add(activity)
            if not activity_resource in unique_activities_resource:
                unique_activities_resource.add(activity_resource)
            count_no_resource += 1
            key = '{}-{}'.format(activity[0], activity[1])
            if key in activities_stats.keys():
                if 'None' in activities_stats[key].keys():
                    activities_stats[key]['None'] += 1
                else:
                    activities_stats[key]['None'] = 1
            else:
                activities_stats[key] = {'None':1}
            if 'None' in resources_stats.keys():
                if key in resources_stats['None'].keys():
                    resources_stats['None'][key] += 1
                else:
                    resources_stats['None'][key] = 1
            else:
                resources_stats['None'] = {key:1}
        if not activity in unique_activities:
            unique_activities.add(activity)

#print(resources)
#print(no_resources)
#print(unique_activities)
#print(unique_activities_resource)

print('Statistics:')
print('Number of unique events: {}'.format(len(unique_activities_resource)))
print('Number of record without resource: {}'.format(count_no_resource))

with open('activities.stat', 'w') as f:
    f.write('----------------------')
    f.write('----  ACTIVITIES  ----')
    f.write('----------------------\n')
    for activity in activities_stats.keys():
        tot_occ = 0
        for res in activities_stats[activity].keys():
            tot_occ += activities_stats[activity][res]
        f.write('*** {} ***\n'.format(activity))
        for res in activities_stats[activity].keys():
            f.write('- {}: {:0.4f}\n'.format(res, activities_stats[activity][res]/tot_occ))

with open('resources.stat', 'w') as f:
    f.write('---------------------')
    f.write('----  RESOURCES  ----')
    f.write('---------------------\n')
    for res in resources_stats.keys():
        tot_occ = 0
        for activity in resources_stats[res].keys():
            tot_occ += resources_stats[res][activity]
        f.write('*** {} ***\n'.format(res))
        for activity in resources_stats[res].keys():
            f.write('- {}: {:0.4f}\n'.format(activity, resources_stats[res][activity]/tot_occ))

# ORGANIZATIONAL MINING
roles = roles_discovery.apply(log)

# similar activities
ja_values = sna.apply(log, variant=sna.Variants.JOINTACTIVITIES_LOG)
gviz_ja_py = sna_visualizer.apply(ja_values, variant=sna_visualizer.Variants.PYVIS)
sna_visualizer.view(gviz_ja_py, variant=sna_visualizer.Variants.PYVIS)

# Handover of work
hw_values = sna.apply(log, variant=sna.Variants.HANDOVER_LOG)
gviz_hw_py = sna_visualizer.apply(hw_values, variant=sna_visualizer.Variants.PYVIS)
sna_visualizer.view(gviz_hw_py, variant=sna_visualizer.Variants.PYVIS)

# Subcontracting
sub_values = sna.apply(log, variant=sna.Variants.SUBCONTRACTING_LOG)
gviz_sub_py = sna_visualizer.apply(sub_values, variant=sna_visualizer.Variants.PYVIS)
sna_visualizer.view(gviz_sub_py, variant=sna_visualizer.Variants.PYVIS)

# Working together
wt_values = sna.apply(log, variant=sna.Variants.WORKING_TOGETHER_LOG)
gviz_sub_py = sna_visualizer.apply(sub_values, variant=sna_visualizer.Variants.PYVIS)
sna_visualizer.view(gviz_sub_py, variant=sna_visualizer.Variants.PYVIS)
