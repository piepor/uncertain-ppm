"""bpic2013 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import pm4py


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for bpic2013 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(bpic2013): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.Sequence({
            'activity': tfds.features.Text(),
            'resource': tfds.features.Text(),
            'impact': tfds.features.Text(),
            'relative_time': tf.int32,
            'day_part': tfds.features.ClassLabel(num_classes=3),
            'week_day': tfds.features.ClassLabel(num_classes=8),
            'case_id': tfds.features.Text()
            }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='http://www.win.tue.nl/bpi/2013/challenge',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(bpic2013): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')

    event_log = pm4py.read_xes('../../BPI_Challenge_2013_incidents.xes')

    return {
        'train': self._generate_examples(event_log),
    }
    # TODO(bpic2013): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'train_imgs'),
    }

  def _generate_examples(self, log):
    """Yields examples."""
    for trace in log:
        case = trace.attributes['concept:name'].split("1-")[1]
        preprocessed_trace = preprocess_trace(trace)
        yield case, {'activity': preprocessed_trace['activity'],
                     'resource': preprocessed_trace['resource'],
                     'impact': preprocessed_trace['impact'],
                     'relative_time': preprocessed_trace['relative_time'],
                     'day_part': preprocessed_trace['day_part'],
                     'week_day': preprocessed_trace['week_day'],
                     'case_id': preprocessed_trace['case_id']}

def preprocess_trace(trace):
    """
    Preprocess events in traces.
    Extract classes from names and computes relative time.
    """
    trace_properties = {'activity': ['<START>'], 'resource': ['1'], 'relative_time': [0], 
                        'impact': [trace[0]['impact']], 'day_part': [0], 'week_day':[0]}

    for count, event in enumerate(trace):
        if count == 0:
            event_date = event['time:timestamp']
            trace_properties['case_id'] = [trace.attributes['concept:name'].split("1-")[1]]
            #trace_properties['day_part'] = [int(event['time:timestamp'].hour > 13)+1]
            #trace_properties['week_day'] = [event['time:timestamp'].isoweekday()]
        trace_properties['case_id'].append(trace.attributes['concept:name'].split("1-")[1])
        delta_t = event['time:timestamp'] - event_date
        event_date = event['time:timestamp']
        trace_properties['activity'].append('{}_{}'.format(event['concept:name'], 
                                                           event['lifecycle:transition']))
        try:
            res = event['org:role']
        except:
            res = '<UNK>'
        trace_properties['resource'].append(res)
        trace_properties['relative_time'].append(delta_t.seconds)
        trace_properties['impact'].append(event['impact'])
        trace_properties['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
        trace_properties['week_day'].append(event['time:timestamp'].isoweekday())

    trace_properties['activity'].append('<END>')
    trace_properties['resource'].append('1')
    trace_properties['impact'].append(event['impact'])
    trace_properties['relative_time'].append(0)
    trace_properties['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
    trace_properties['week_day'].append(event['time:timestamp'].isoweekday())
    trace_properties['case_id'].append(trace.attributes['concept:name'].split("1-")[1])

    return trace_properties
