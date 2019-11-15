import pandas as pd
import os
from sklearn.utils import shuffle

SPEECH_COMMAND_DIR = 'data/sd_GSCmdV2/train/'

words = os.listdir(SPEECH_COMMAND_DIR)

def filter_files(files):
    unfiles = [
        'LICENSE', 'README.md', 'validation_list.txt', 'testing_list.txt',
        '.DS_Store', '_background_noise_'
    ]
    return [f for f in files if f not in unfiles]

words = filter_files(words)
wavs = []
texts = []
filesizes = []
for word in words:
    word_dir = SPEECH_COMMAND_DIR + word + '/'
    for wav in os.listdir(word_dir):
        wavs.append(word_dir + wav)
        texts.append(word)
        filesizes.append(os.path.getsize(word_dir + wav))

df = pd.DataFrame({
    'wav_filename': wavs, 'wav_filesize': filesizes, 'transcript': texts
})

df = shuffle(df)

test_split = 0.8
validation_split = 0.9
test_split_no = int(len(df) * test_split)
validation_split_no = int(len(df) * validation_split)
df[:test_split_no].to_csv('gsc-train.csv')
df[test_split_no:validation_split_no:].to_csv('gsc-test.csv')
df[validation_split_no:].to_csv('gsc-valid.csv')