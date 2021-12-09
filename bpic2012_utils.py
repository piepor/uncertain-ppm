import tensorflow as tf

def bpic2012_utils(tfds_id):
    padded_shapes = {
        'activity': [None],
        'resource': [None],
        'amount': [None],
        'relative_time': [None],
        'day_part': [None],
        'week_day': [None],
        'case_id': [None],
    }
    padding_values = {
        'activity': '<PAD>',
        'resource': '<PAD>',
        'amount': tf.cast(0, dtype=tf.int32),
        'relative_time': tf.cast(0, dtype=tf.int32),
        'day_part': tf.cast(0, dtype=tf.int64),
        'week_day': tf.cast(0, dtype=tf.int64),
        'case_id': '<PAD>',
    }
    if tfds_id:
        padded_shapes['tfds_id'] = ()
        padding_values['tfds_id'] = '<PAD>'
    vocabulary_act = ['<START>',  '<END>', 'A_CANCELLED_COMPLETE', 'A_DECLINED_COMPLETE', 
                      'A_APPROVED_COMPLETE', 'A_REGISTERED_COMPLETE', 'A_ACCEPTED_COMPLETE',
                      'A_PARTLYSUBMITTED_COMPLETE', 'A_FINALIZED_COMPLETE', 'A_PREACCEPTED_COMPLETE',
                      'A_ACTIVATED_COMPLETE', 'A_SUBMITTED_COMPLETE', 'O_DECLINED_COMPLETE',
                      'O_SENT_COMPLETE', 'O_ACCEPTED_COMPLETE', 'O_CANCELLED_COMPLETE',
                      'O_SENT_BACK_COMPLETE', 'O_SELECTED_COMPLETE', 'O_CREATED_COMPLETE',
                      'W_Nabellen offertes_SCHEDULE', 'W_Nabellen offertes_START', 'W_Nabellen offertes_COMPLETE', 
                      'W_Nabellen incomplete dossiers_SCHEDULE', 'W_Nabellen incomplete dossiers_START', 
                      'W_Nabellen incomplete dossiers_COMPLETE', 
                      'W_Beoordelen fraude_SCHEDULE', 'W_Beoordelen fraude_START', 'W_Beoordelen fraude_COMPLETE', 
                      'W_Wijzigen contractgegevens_SCHEDULE',
                      'W_Afhandelen leads_START', 'W_Afhandelen leads_SCHEDULE', 'W_Afhandelen leads_COMPLETE',
                      'W_Valideren aanvraag_SCHEDULE', 'W_Valideren aanvraag_START', 'W_Valideren aanvraag_COMPLETE',
                      'W_Completeren aanvraag_SCHEDULE', 'W_Completeren aanvraag_START', 'W_Completeren aanvraag_COMPLETE',
                      ] 

    vocabulary_res = ['1', '11302', '10931', '10982', '10609', '11049', '11079', '112', '11181',
                      '11179', '11180', '10188', '10899', '10880', '11002', '11111', '<UNK>',
                      '10138', '11202', '11119', '10881', '11259', '10932', '11304', '11120',
                      '11019', '11122', '10971', '10228', '10935', '10909', '11201', '11309',
                      '10125', '10862', '11300', '10914', '11339', '11289', '10889', '10910', 
                      '10859', '10972', '11299', '10809', '11189', '10629', '11121', '11200',
                      '11000', '11001', '10861', '10929', '10779', '11254', '11203', '10789',
                      '10124', '10821', '11169', '11319', '11009', '11269', '10863', '11029',
                      '10939', '11003', '10933', '10912', '10913']

    features_type = {'org:resource': 'categorical', 'AMOUNT_REQ': 'continuous',
                     'day_part': 'categorical', 'week_day': 'categorical'}

    features_variant = {'org:resource':'event', 'AMOUNT_REQ': 'trace'}

    return padded_shapes, padding_values, vocabulary_act, vocabulary_res, features_type, features_variant
