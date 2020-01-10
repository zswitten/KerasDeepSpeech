import numpy as np
from generator import make_mfcc_shape, get_intseq

def make_query(audio_files, query, model_input_type='mfcc', max_len=1000):
    if model_input_type != 'mfcc':
        return 'not implemented'
    X_data = np.array([make_mfcc_shape(file_name, padlen=max_len) for file_name in audio_files])
    queries_words = [query] * len(audio_files)
    max_query_len = 30
    queries = np.array([get_intseq(query, max_query_len)] * len(audio_files))
    source_str = ''
    inputs = {
        'the_input': X_data,
        'query_words': queries_words,
        'query': queries,
        'source_str': source_str
    }

    outputs = {'y_pred': [None] * len(audio_files)}

    return (inputs, outputs)

def old_make_query(audio_file, query, model_input_type='mfcc', max_len=1000):
    if model_input_type != 'mfcc':
        return 'not implemented'
    X_data = np.array([make_mfcc_shape(file_name, padlen=max_len) for file_name in [audio_file]])
    queries_words = [query]
    max_query_len = 30
    queries = np.array([get_intseq(query, max_query_len) for query in queries_words])
    source_str = ''
    inputs = {
        'the_input': X_data,
        'query_words': queries_words,
        'query': queries,
        'source_str': source_str
    }

    outputs = {'y_pred': [True]}

    return (inputs, outputs)
