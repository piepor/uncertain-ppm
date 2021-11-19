"""bpic2015 dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import os
import shutil
import pm4py

# TODO(bpic2015): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(bpic2015): BibTeX citation
_CITATION = """
"""

VERSION = tfds.core.Version('1.0.0')

class bpic2015Config(tfds.core.BuilderConfig):
    """Builder config for bpic2015"""
    def __init__(self, *, municipality=None, **kwargs):
        super(bpic2015Config, self).__init__(version=VERSION, **kwargs)
        self.municipality = municipality

class Bpic2015(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for bpic2015 dataset."""

  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      bpic2015Config(
          name='m1', description='Municipality 1', municipality='Municipality1'),
      bpic2015Config(
          name='m2', description='Municipality 2', municipality='Municipality2'),
      bpic2015Config(
          name='m3', description='Municipality 3', municipality='Municipality3'),
  ]
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
            features=tfds.features.Sequence({
                'activity': tfds.features.Text(),
                'resource': tfds.features.Text(),
                'relative_time': tf.int32,
                'day_part': tfds.features.ClassLabel(num_classes=3),
                'week_day': tfds.features.ClassLabel(num_classes=8)
                }),

            supervised_keys=None,  # Set to `None` to disable
            homepage='http://www.win.tue.nl/bpi/2015/challenge',
            citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # Downloads the data and defines the splits
    #breakpoint()
    path1 = dl_manager.download_and_extract('https://data.4tu.nl/ndownloader/files/24063818')
    path2 = dl_manager.download_and_extract('https://data.4tu.nl/ndownloader/files/24044639')
    path3 = dl_manager.download_and_extract('https://data.4tu.nl/ndownloader/files/24076154')
    path4 = dl_manager.download_and_extract('https://data.4tu.nl/ndownloader/files/24045332')
    path5 = dl_manager.download_and_extract('https://data.4tu.nl/ndownloader/files/24069341')
    try:
        os.mkdir(os.path.join(path1.parent, 'extracted', 'Municipality1'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path2.parent, 'extracted', 'Municipality2'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path3.parent, 'extracted', 'Municipality3'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path4.parent, 'extracted', 'Municipality4'))
    except:
        pass
    try:
        os.mkdir(os.path.join(path5.parent, 'extracted', 'Municipality5'))
    except:
        pass
    new_path = path1.parent / 'extracted' / 'Municipality1' / 'event_log.xes'
    shutil.copy(path1.as_posix(), new_path.as_posix())
    new_path = path1.parent / 'extracted' / 'Municipality2' / 'event_log.xes'
    shutil.copy(path2.as_posix(), new_path.as_posix())
    new_path = path1.parent / 'extracted' / 'Municipality3' / 'event_log.xes'
    shutil.copy(path3.as_posix(), new_path.as_posix())
    new_path = path1.parent / 'extracted' / 'Municipality4' / 'event_log.xes'
    shutil.copy(path4.as_posix(), new_path.as_posix())
    new_path = path1.parent / 'extracted' / 'Municipality5' / 'event_log.xes'
    shutil.copy(path5.as_posix(), new_path.as_posix())
    return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, 
                                    gen_kwargs={
                                        "filepath":
                                        os.path.join(path1.parent, 
                                                     'extracted',
                                                     self._builder_config.municipality)
                                    })]
    # TODO(bpic2015): Returns the Dict[split names, Iterator[Key, Example]]
#    return {
#        'train': self._generate_examples(path / 'train_imgs'),
#    }
    

  def _generate_examples(self, filepath):
    """Yields examples."""
    # TODO(bpic2015): Yields (key, example) tuples from the dataset
#    for f in path.glob('*.jpeg'):
#      yield 'key', {
#          'image': f,
#          'label': 'yes',
    #}
    event_log = pm4py.read_xes(os.path.join(
        filepath, 'event_log.xes'))
    for trace in event_log:
        case = trace.attributes['concept:name']
        preprocessed_trace = preprocess_trace(trace)
        yield case, {'activity': preprocessed_trace['activity'],
                     'resource': preprocessed_trace['resource'],
                     'relative_time': preprocessed_trace['relative_time'],
                     'day_part': preprocessed_trace['day_part'],
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
            trace_properties['day_part'] = [int(event['time:timestamp'].hour > 13)+1]
            trace_properties['week_day'] = [event['time:timestamp'].isoweekday()]
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

    return trace_properties
