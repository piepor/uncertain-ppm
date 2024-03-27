"""road_traffic_fines dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import pm4py
import copy


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for road_traffic_fines dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(road_traffic_fines): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.Sequence({
            'activity': tfds.features.Text(),
            'resource': tfds.features.Text(),
            'article': tfds.features.Text(),
            'amount': tf.float32,
            'totalPaymentAmount': tf.float32,
            'points': tf.int32,
            'relative_time': tf.int32,
            'day_part': tfds.features.ClassLabel(num_classes=3),
            'week_day': tfds.features.ClassLabel(num_classes=8),
            'case_id': tfds.features.Text()
            }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://data.4tu.nl/articles/dataset/Road_Traffic_Fine_Management_Process/12683249',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(road_traffic_fines): Downloads the data and defines the splits
    event_log = pm4py.read_xes('../../log-Road_Traffic_Fine_Management_Process.xes')

    return {
        'train': self._generate_examples(event_log),
    }

  def _generate_examples(self, log):
    """Yields examples."""
    for cont, trace in enumerate(log):
        case = str(cont)
        preprocessed_trace = preprocess_trace(trace, str(cont))
        yield case, {'activity': preprocessed_trace['activity'],
                     'resource': preprocessed_trace['resource'],
                     'amount': preprocessed_trace['amount'],
                     'article': preprocessed_trace['article'],
                     'points': preprocessed_trace['points'],
                     'totalPaymentAmount': preprocessed_trace['totalPaymentAmount'],
                     'relative_time': preprocessed_trace['relative_time'],
                     'day_part': preprocessed_trace['day_part'],
                     'week_day': preprocessed_trace['week_day'],
                     'case_id': preprocessed_trace['case_id']}

def preprocess_trace(trace, case_id):
    """
    Preprocess events in traces.
    Extract classes from names and computes relative time.
    """
    trace_properties = {'activity': ['<START>'], 'resource': ['1'],
                        'relative_time': [0], 'article': ['0'], 'points': [0], 'totalPaymentAmount': [0],
                        'amount': [0], 'day_part': [0], 'week_day':[0]}
    totalPaymentAmount = 0
    amount = 0
    article = str(trace[0]['article'])
    points = trace[0]['points']
    for count, event in enumerate(trace):
        if count == 0:
            event_date = event['time:timestamp']
            trace_properties['case_id'] = [case_id]
            #trace_properties['day_part'] = [int(event['time:timestamp'].hour > 13)+1]
            #trace_properties['week_day'] = [event['time:timestamp'].isoweekday()]
        trace_properties['case_id'].append(case_id)
        delta_t = event['time:timestamp'] - event_date
        event_date = event['time:timestamp']
        trace_properties['activity'].append(f"{event['concept:name']}")
        try:
            res = event['org:role']
        except:
            res = '<UNK>'
        trace_properties['resource'].append(res)
        if 'amount' in event:
            amount = event['amount']
        trace_properties['amount'].append(amount)
        if 'totalPaymentAmount' in event:
            totalPaymentAmount = event['totalPaymentAmount']
        trace_properties['totalPaymentAmount'].append(totalPaymentAmount)
        if 'points' in event:
            points = event['points']
        trace_properties['points'].append(points)
        if 'article' in event:
            article = str(event['article'])
        trace_properties['article'].append(article)
        trace_properties['relative_time'].append(delta_t.seconds)
        trace_properties['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
        trace_properties['week_day'].append(event['time:timestamp'].isoweekday())

    trace_properties['activity'].append('<END>')
    trace_properties['resource'].append('1')
    trace_properties['amount'].append(amount)
    trace_properties['totalPaymentAmount'].append(totalPaymentAmount)
    trace_properties['article'].append(article)
    trace_properties['points'].append(points)
    trace_properties['relative_time'].append(0)
    trace_properties['day_part'].append(int(event['time:timestamp'].hour > 13)+1)
    trace_properties['week_day'].append(event['time:timestamp'].isoweekday())
    trace_properties['case_id'].append(case_id)

    return trace_properties
