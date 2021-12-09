"""bpic2012 dataset."""

import tensorflow_datasets as tfds
import pm4py
import tensorflow as tf
import os

# TODO(bpic2012): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Event log from a Dutch Financial Institute regarding the application process for a personal loan or overdraft.
It contains:
262.200 events
13.087 cases
36 activities (+3 for start, end and padding)

*Dataset Features*

* activity: name of the activity [39 different possibilities]
* resource: who carried on the activity [68 different possibilities + <UNK> + start and end]
* amount: amount of money requested
* relative_time: time passed from the last activity [seconds]
* day_part: if the event happend before 13.00 [class 1] or after [class 2]
* week_day: day of the week [class 1 = Monday]

All the classes referring to resource, product (ecc...) have been transformed to integer from the 
original string (e.g. 'Value 1' -> 1). 
Class 0  is always reserved for a possible padding.
Relative time between two events has been computed subtracting the original timestamps.

In order for the dataset to be processed [pm4py][1] is required

[1]https://pm4py.fit.fraunhofer.de/
"""

_CITATION = """
1. van Dongen, B.. (2012, April 23). BPI Challenge 2012. 4TU.ResearchData. 
https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f 
"""


class Bpic2012(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for bpic2012 dataset."""

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
                'resource': tfds.features.Text(),
                'amount': tf.int32,
                'relative_time': tf.int32,
                'day_part': tfds.features.ClassLabel(num_classes=3),
                'week_day': tfds.features.ClassLabel(num_classes=8),
                'case_id': tfds.features.Text()
                }),

            supervised_keys=None,  # Set to `None` to disable
            homepage='http://www.win.tue.nl/bpi/2012/challenge',
            citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract({'file':'https://data.4tu.nl/ndownloader/articles/12689204/versions/1'})
    event_log = pm4py.read_xes(os.path.join(path['file'].as_posix(), 'BPI_Challenge_2012.xes.gz'))

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
                     'amount': preprocessed_trace['amount'],
                     'relative_time': preprocessed_trace['relative_time'],
                     'day_part': preprocessed_trace['day_part'],
                     'case_id': preprocessed_trace['case_id']}
                     'week_day': preprocessed_trace['week_day']}

def preprocess_trace(trace):
    """
    Preprocess events in traces.
    Extract classes from names and computes relative time.
    """
    trace_properties = {'activity': ['<START>'], 'resource': ['1'], 'relative_time': [0], 
                        'amount': [int(trace.attributes['AMOUNT_REQ'])], 'day_part': [0], 'week_day':[0]}

    for count, event in enumerate(trace):
        if count == 0:
            event_date = event['time:timestamp']
            trace_properties['case_id'] = trace.attributes['concept:name']
            #trace_properties['day_part'] = [int(event['time:timestamp'].hour > 13)+1]
            #trace_properties['week_day'] = [event['time:timestamp'].isoweekday()]
        trace_properties['case_id'].append(trace.attributes['concept:name'])
        delta_t = event['time:timestamp'] - event_date
        event_date = event['time:timestamp']
        trace_properties['activity'].append('{}_{}'.format(event['concept:name'], 
                                                           event['lifecycle:transition']))
        try:
            res = event['org:resource']
        except:
            res = '<UNK>'
        trace_properties['resource'].append(res)
        trace_properties['relative_time'].append(delta_t.seconds)
        trace_properties['amount'].append(int(trace.attributes['AMOUNT_REQ']))
        trace_properties['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
        trace_properties['week_day'].append(event['time:timestamp'].isoweekday())

    trace_properties['activity'].append('<END>')
    trace_properties['resource'].append('1')
    trace_properties['amount'].append(int(trace.attributes['AMOUNT_REQ']))
    trace_properties['relative_time'].append(0)
    trace_properties['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
    trace_properties['week_day'].append(event['time:timestamp'].isoweekday())
    trace_properties['case_id'].append(trace.attributes['concept:name'])

    return trace_properties
