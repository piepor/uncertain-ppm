"""helpdesk dataset."""

import tensorflow_datasets as tfds
import tensorflow as tf
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
import datetime
import pandas as pd

# TODO(helpdesk): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """

Event log concerning the ticketing management process of the Help desk
of an Italian software company.

Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(helpdesk): BibTeX citation
_CITATION = """
"""


class Helpdesk(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for helpdesk dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
    Register into https://example.org/login to get the data. Place the `data.zip`
    file in the `manual_dir/`.
    """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(helpdesk): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        #features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
#            'image': tfds.features.Image(shape=(None, None, 3)),
#            'label': tfds.features.ClassLabel(names=['no', 'yes']),
            #'case': tfds.features.Text(),
        features=tfds.features.Sequence({
                'activity': tfds.features.Text(),
                'resource': tf.int32,
                'product': tf.int32,    
                'costumer': tf.int32,    
                'responsible_section': tf.int32,    
                'service_level': tf.int32,    
                'service_type': tf.int32,    
                'seriousness': tf.int32,    
                'workgroup': tf.int32,
                'variant': tf.int32,    
                'deltatime': tf.int32,
            }),
        #}),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        #supervised_keys=('image', 'label'),  # Set to `None` to disable
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://data.4tu.nl/ndownloader/files/23993303',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(helpdesk): Downloads the data and defines the splits
    #path = dl_manager.download_and_extract('https://todo-data-url')
    path = dl_manager.manual_dir / 'Helpdesk.xes'
    path = '../../finale.csv'
    log_csv = pd.read_csv(path, sep=',')
    log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
    log_csv = log_csv.sort_values('Complete Timestamp')
    log_csv.rename(columns={'Case ID': 'case:concept:name', 
                            'Complete Timestamp': 'time:timestamp'}, inplace=True)
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
                  log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case:'}
    event_log = log_converter.apply(log_csv, parameters=parameters,
                                    variant=log_converter.Variants.TO_EVENT_LOG)
    #log_csv = pm4py.read_xes(path)

    # TODO(helpdesk): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        #'train': self._generate_examples(path),
        'train': self._generate_examples(event_log),
    }

  def _generate_examples(self, log):
    """Yields examples."""
    # TODO(helpdesk): Yields (key, example) tuples from the dataset
#    for f in path.glob('*.jpeg'):
#      yield 'key', {
#          'image': f,
#          'label': 'yes',
#      }
    for trace in log:

#      yield 'case', {
        case = trace.attributes['concept:name']
        act = []
        res = []
        #life = []
        time = []
        prod = []
        cost = []
        resp = []
        serv_lev = []
        serv_type = []
        seri = []
        wg = []
        var = []

        for count, event in enumerate(trace):
            if count == 0:
                delta_t = datetime.timedelta(0)
                event_date = event['time:timestamp']
            else:
                delta_t = event['time:timestamp'] - event_date
                event_date = event['time:timestamp']
#            act.append(event['concept:name'])
#            res.append(event['org:resource'].split()[1])
#            life.append(event['lifecycle:transition'])
            act.append(event['Activity'])
            res.append(int(event['Resource'].split()[1]))
            prod.append(int(event['product'].split()[1]))
            cost.append(int(event['customer'].split()[1]))
            resp.append(int(event['responsible_section'].split()[1]))
            serv_lev.append(int(event['service_level'].split()[1]))
            serv_type.append(int(event['service_type'].split()[1]))
            seri.append(int(event['seriousness'].split()[1]))
            wg.append(int(event['workgroup'].split()[1]))
            var.append(event['Variant index'])
            time.append(delta_t.seconds)
         
        yield case, {'activity': act,
                     'resource': res,
                     'product': prod,    
                     'costumer': cost,
                     'responsible_section': resp,    
                     'service_level': serv_lev,    
                     'service_type': serv_type,    
                     'seriousness': seri,    
                     'workgroup': wg,
                     'variant': var,    
                     'deltatime': time}
#        yield case, {
#            'activity': [event['concept:name'] for event in trace],
#            'resource': [event['org:resource'] for event in trace],
#            'lifecycle': [event['lifecycle:transition'] for event in trace],
#            'timestamp': [event['time:timestamp'].minute for event in trace],
#        }
