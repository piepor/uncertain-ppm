import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.util import dataframe_utils
import datetime
import pandas as pd

act = set()
res = set()
prod = set()
cust = set()
resp = set()
serv_lev = set()
serv_type = set()
wg = set()
seri = set()
var = set()

log_csv = pd.read_csv('./dummy_data/finale.csv', sep=',')
# Create the Event Log object as in the library pm4py
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
log_csv = log_csv.sort_values('Complete Timestamp')
log_csv.rename(columns={'Case ID': 'case:concept:name', 
                        'Complete Timestamp': 'time:timestamp'}, inplace=True)
parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name',
              log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ATTRIBUTE_PREFIX: 'case:'}
event_log = log_converter.apply(log_csv, parameters=parameters,
                                variant=log_converter.Variants.TO_EVENT_LOG)

for trace in event_log:
    for event in trace:
        act.add(event['Activity'])
        res.add(event['Resource'])
        cust.add(event['customer'])
        resp.add(event['responsible_section'])
        serv_lev.add(event['service_level'])
        serv_type.add(event['service_type'])
        wg.add(event['workgroup'])
        seri.add(event['seriousness_2'])
        var.add(event['Variant index'])
