"""helpdesk dataset."""

import datetime
import tensorflow_datasets as tfds
import tensorflow as tf
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
import pandas as pd

_DESCRIPTION = """

Event log concerning the ticketing management process of the Help desk
of an Italian software company. It contains:
21384 events
4580 cases
14 activities

*Dataset Features*

* activity: name of the activity [14 different possibilities]
* resource: class representing who carried on the activity [22 different possibilities]
* product: product related to the activity [21 different possibilities]
* customer: class representing the customer [397 different posibilities]
* responsible_section: section responsible for the activity [7 different possibilities]
* service_level: level of the service requested for the activity [4 different possibilities]
* service_type: type of the service requested for the activity [4 different possibilities]
* workgroup: group of the resource [4 different possibilities]
* seriousness: seriousness level of the activity. Sub-seriousness (as called in the [readme.txt][1]) 
has been taken because the main seriousness has only 1 value. [4 different possibilities]
* variant: variant of the trace [226 different possibilities]
* relative_time: time passed from the last activity [seconds]
* day_part: if the event happend before 13.00 [class 1] or after [class 2]
* week_day: day of the week [class 1 = Monday]

All the classes referring to resource, product (ecc...) have been transformed to integer from the 
original string (e.g. 'Value 1' -> 1). 
Class 0  is always reserved for a possible padding.
Relative time between two events has been computed subtracting the original timestamps.

In order for the dataset to be processed [pandas][2] and [pm4py][3] are required

[1]https://data.4tu.nl/articles/dataset/Dataset_belonging_to_the_help_desk_log_of_an_Italian_Company/12675977?file=23993306
[2]https://pandas.pydata.org/
[3]https://pm4py.fit.fraunhofer.de/
"""

_CITATION = """
Polato, Mirko (2017): Dataset belonging to the help desk log of an Italian Company. 4TU.ResearchData. 
Dataset. https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb 
"""

class Helpdesk(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for helpdesk dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.Sequence({
                'activity': tfds.features.Text(),
                'resource': tfds.features.ClassLabel(num_classes=24),
                'product': tfds.features.ClassLabel(num_classes=23),
                'customer': tfds.features.ClassLabel(num_classes=398),
                'responsible_section': tfds.features.ClassLabel(num_classes=9),
                'service_level': tfds.features.ClassLabel(num_classes=6),
                'service_type': tfds.features.ClassLabel(num_classes=6),
                'workgroup': tfds.features.ClassLabel(num_classes=6),
                'seriousness': tfds.features.ClassLabel(num_classes=6),
                'variant': tfds.features.ClassLabel(num_classes=227),
                'relative_time': tf.int32,
                'day_part': tfds.features.ClassLabel(num_classes=3),
                'week_day': tfds.features.ClassLabel(num_classes=8)
                }),

            supervised_keys=None,  # Set to `None` to disable
            homepage='https://data.4tu.nl/ndownloader/files/23993303',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Downloads the data and reads the csv file
        path = dl_manager.download({'file':'https://data.4tu.nl/ndownloader/files/23993303'})
        log_csv = pd.read_csv(path['file'] , sep=',')
        # Create the Event Log object as in the library pm4py
        log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
        log_csv = log_csv.sort_values('Complete Timestamp')
        log_csv.rename(columns={'Case ID': 'case:concept:name',
                                'Complete Timestamp': 'time:timestamp'}, inplace=True)
        parameters = {
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
            log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case:'}
        event_log = log_converter.apply(log_csv, parameters=parameters,
                                        variant=log_converter.Variants.TO_EVENT_LOG)

        return {
            'train': self._generate_examples(event_log),
        }


    def _generate_examples(self, log):
        """Yields examples."""
        for trace in log:
            case = trace.attributes['concept:name']
            preprocessed_trace = preprocess_trace(trace)
            yield case, {'activity': preprocessed_trace['activity'],
                         'resource': preprocessed_trace['resource'],
                         'product': preprocessed_trace['product'],
                         'customer': preprocessed_trace['customer'],
                         'responsible_section': preprocessed_trace['responsible_section'],
                         'service_level': preprocessed_trace['service_level'],
                         'service_type': preprocessed_trace['service_type'],
                         'workgroup': preprocessed_trace['workgroup'],
                         'seriousness': preprocessed_trace['seriousness'],
                         'variant': preprocessed_trace['variant'],
                         'relative_time': preprocessed_trace['relative_time'],
                         'day_part': preprocessed_trace['day_part'],
                         'week_day': preprocessed_trace['week_day']}

def preprocess_trace(trace):
    """
    Preprocess events in traces.
    Extract classes from names and computes relative time.
    """
    trace_properties = {'activity': ['<START>'], 'resource': [1], 'relative_time': [0], 'product': [1],
                        'customer': [1], 'responsible_section': [1], 'service_level': [1],
                        'service_type': [1], 'workgroup': [1], 'variant': [1], 'seriousness': [1],
                        'day_part': [0], 'week_day':[0]}
    #trace_properties = dict.fromkeys(keys)

    for count, event in enumerate(trace):
        if count == 0:
#            delta_t = datetime.timedelta(0)
            event_date = event['time:timestamp']
            trace_properties['day_part'] = [int(event['time:timestamp'].hour > 13)+1]
            trace_properties['week_day'] = [event['time:timestamp'].isoweekday()]
            trace_properties['variant'] = [event['Variant index']]
            trace_properties['customer'] = [int(event['customer'].split()[1])]
#            trace_properties['activity'] = [event['Activity']]
#            trace_properties['resource'] = [int(event['Resource'].split()[1])]
#            trace_properties['product'] = [int(event['product'].split()[1])]
#            trace_properties['customer'] = [int(event['customer'].split()[1])]
#            trace_properties['responsible_section'] = [int(event['responsible_section'].split()[1])]
#            trace_properties['service_level'] = [int(event['service_level'].split()[1])]
#            trace_properties['service_type'] = [int(event['service_type'].split()[1])]
#            trace_properties['workgroup'] = [int(event['workgroup'].split()[1])]
#            trace_properties['variant'] = [event['Variant index']]
#            trace_properties['seriousness'] = [int(event['seriousness_2'].split()[1])]
#            trace_properties['relative_time'] = [delta_t.seconds]
#            trace_properties['day_part'] = [int(event['time:timestamp'].hour > 13)+1]
#            trace_properties['week_day'] = [event['time:timestamp'].isoweekday()]
        else:
            delta_t = event['time:timestamp'] - event_date
            event_date = event['time:timestamp']
            trace_properties['activity'].append(event['Activity'])
            trace_properties['resource'].append(int(event['Resource'].split()[1])+1)
            trace_properties['product'].append(int(event['product'].split()[1])+1)
            trace_properties['customer'].append(int(event['customer'].split()[1]))
            trace_properties['responsible_section'].append(
                int(event['responsible_section'].split()[1])+1)
            trace_properties['service_level'].append(int(event['service_level'].split()[1])+1)
            trace_properties['service_type'].append(int(event['service_type'].split()[1])+1)
            trace_properties['workgroup'].append(int(event['workgroup'].split()[1])+1)
            trace_properties['variant'].append(event['Variant index'])
            trace_properties['seriousness'].append(int(event['seriousness_2'].split()[1])+1)
            trace_properties['relative_time'].append(delta_t.seconds)
            trace_properties['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
            trace_properties['week_day'].append(event['time:timestamp'].isoweekday())
            event_date = event['time:timestamp']

    trace_properties['activity'].append('<END>')
    trace_properties['resource'].append(1)
    trace_properties['product'].append(1)
    trace_properties['customer'].append(1)
    trace_properties['responsible_section'].append(1)
    trace_properties['service_level'].append(1)
    trace_properties['service_type'].append(1)
    trace_properties['workgroup'].append(1)
    trace_properties['variant'].append(event['Variant index'])
    trace_properties['seriousness'].append(int(event['seriousness_2'].split()[1]))
    trace_properties['relative_time'].append(0)
    trace_properties['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
    trace_properties['week_day'].append(event['time:timestamp'].isoweekday())

    return trace_properties
