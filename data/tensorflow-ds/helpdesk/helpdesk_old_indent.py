"""helpdesk dataset."""

import datetime
import tensorflow_datasets as tfds
import tensorflow as tf
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
import pandas as pd

_DESCRIPTION = """

Event log concerning the ticketing management process of the Help desk
of an Italian software company. It contains
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
* deltatime: time passed from the last activity [seconds]

All the classes referring to resource, product (ecc...) have been transformed to integer from the 
original string (e.g. 'Value 1' -> 1). Class 0  is reserved for a possible padding.
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
                'resource': tfds.features.ClassLabel(num_classes=23),
                'product': tfds.features.ClassLabel(num_classes=22),    
                'customer': tfds.features.ClassLabel(num_classes=398),    
                'responsible_section': tfds.features.ClassLabel(num_classes=8),    
                'service_level': tfds.features.ClassLabel(num_classes=5),    
                'service_type': tfds.features.ClassLabel(num_classes=5),    
                'workgroup': tfds.features.ClassLabel(num_classes=5),
                'seriousness': tfds.features.ClassLabel(num_classes=5),
                'variant': tfds.features.ClassLabel(num_classes=227),    
                'deltatime': tf.int32,
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
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
                  log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case:'}
    event_log = log_converter.apply(log_csv, parameters=parameters,
                                    variant=log_converter.Variants.TO_EVENT_LOG)

    return {
        'train': self._generate_examples(event_log),
    }
 
  def _preprocess_trace(self, trace):
    """ 
    Preprocess events in traces.
    Extract classes from names and computes relative time.
    """
    act = []
    res = []
    time = []
    prod = []
    cust = []
    resp = []
    serv_lev = []
    serv_type = []
    wg = []
    seri = []
    var = []

    for count, event in enumerate(trace):
        if count == 0:
            delta_t = datetime.timedelta(0)
            event_date = event['time:timestamp']
        else:
            delta_t = event['time:timestamp'] - event_date
            event_date = event['time:timestamp']
        act.append(event['Activity'])
        res.append(int(event['Resource'].split()[1]))
        prod.append(int(event['product'].split()[1]))
        cust.append(int(event['customer'].split()[1]))
        resp.append(int(event['responsible_section'].split()[1]))
        serv_lev.append(int(event['service_level'].split()[1]))
        serv_type.append(int(event['service_type'].split()[1]))
        wg.append(int(event['workgroup'].split()[1]))
        seri.append(int(event['seriousness_2'].split()[1]))
        var.append(event['Variant index'])
        time.append(delta_t.seconds)
    
    return act, res, prod, cust, resp, serv_lev, serv_type, wg, seri, var, time

  def _generate_examples(self, log):
    """Yields examples."""
    for trace in log:
        case = trace.attributes['concept:name']
        preprocessed_trace = self._preprocess_trace(trace) 
        yield case, {'activity': preprocessed_trace[0],
                     'resource': preprocessed_trace[1],
                     'product': preprocessed_trace[2],    
                     'customer': preprocessed_trace[3],
                     'responsible_section': preprocessed_trace[4],    
                     'service_level': preprocessed_trace[5],    
                     'service_type': preprocessed_trace[6],    
                     'workgroup': preprocessed_trace[7],
                     'seriousness': preprocessed_trace[8],
                     'variant': preprocessed_trace[9],    
                     'deltatime': preprocessed_trace[10]}
