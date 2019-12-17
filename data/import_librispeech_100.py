from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import codecs
import fnmatch
import pandas
import subprocess
import tarfile
import unicodedata

from sox import Transformer
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import gfile

def _download_and_preprocess_data(data_dir):
    # Conditionally download data to data_dir
    print("Downloading Librivox data set (55GB) into {} if not already present...".format(data_dir))
    TRAIN_CLEAN_100_URL = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"

    DEV_CLEAN_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
    DEV_OTHER_URL = "http://www.openslr.org/resources/12/dev-other.tar.gz"

    TEST_CLEAN_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
    TEST_OTHER_URL = "http://www.openslr.org/resources/12/test-other.tar.gz"

    def filename_of(x): return os.path.split(x)[1]
    train_clean_100 = base.maybe_download(filename_of(TRAIN_CLEAN_100_URL), data_dir, TRAIN_CLEAN_100_URL)

    dev_clean = base.maybe_download(filename_of(DEV_CLEAN_URL), data_dir, DEV_CLEAN_URL)
    dev_other = base.maybe_download(filename_of(DEV_OTHER_URL), data_dir, DEV_OTHER_URL)

    test_clean = base.maybe_download(filename_of(TEST_CLEAN_URL), data_dir, TEST_CLEAN_URL)
    test_other = base.maybe_download(filename_of(TEST_OTHER_URL), data_dir, TEST_OTHER_URL)

    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    print("Extracting librivox data if not already extracted...")
    LIBRIVOX_DIR = "LibriSpeech"
    work_dir = os.path.join(data_dir, LIBRIVOX_DIR)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "train-clean-100"), train_clean_100)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-clean"), dev_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "dev-other"), dev_other)

    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-clean"), test_clean)
    _maybe_extract(data_dir, os.path.join(LIBRIVOX_DIR, "test-other"), test_other)

    # Convert FLAC data to wav, from:
    #  data_dir/LibriSpeech/split/1/2/1-2-3.flac
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-3.wav
    #
    # And split LibriSpeech transcriptions, from:
    #  data_dir/LibriSpeech/split/1/2/1-2.trans.txt
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-0.txt
    #  data_dir/LibriSpeech/split-wav/1-2-1.txt
    #  data_dir/LibriSpeech/split-wav/1-2-2.txt
    #  ...
    print("Converting FLAC to WAV and splitting transcriptions...")
    train_100 = _convert_audio_and_split_sentences(work_dir, "train-clean-100", "train-clean-100-wav")

    dev_clean = _convert_audio_and_split_sentences(work_dir, "dev-clean", "dev-clean-wav")
    dev_other = _convert_audio_and_split_sentences(work_dir, "dev-other", "dev-other-wav")

    test_clean = _convert_audio_and_split_sentences(work_dir, "test-clean", "test-clean-wav")
    test_other = _convert_audio_and_split_sentences(work_dir, "test-other", "test-other-wav")

    # Write sets to disk as CSV files
    train_100.to_csv(os.path.join(data_dir, "librivox-train-clean-100.csv"), index=False)

    dev_clean.to_csv(os.path.join(data_dir, "librivox-dev-clean.csv"), index=False)
    dev_other.to_csv(os.path.join(data_dir, "librivox-dev-other.csv"), index=False)

    test_clean.to_csv(os.path.join(data_dir, "librivox-test-clean.csv"), index=False)
    test_other.to_csv(os.path.join(data_dir, "librivox-test-other.csv"), index=False)

def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()

def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir):
    source_dir = os.path.join(extracted_dir, data_set)
    target_dir = os.path.join(extracted_dir, dest_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    files = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, '*.trans.txt'):
            trans_filename = os.path.join(root, filename)
            with codecs.open(trans_filename, "r", "utf-8") as fin:
                for line in fin:
                    # Parse each segment line
                    first_space = line.find(" ")
                    seqid, transcript = line[:first_space], line[first_space+1:]

                    transcript = unicodedata.normalize("NFKD", transcript)  \
                                            .encode("ascii", "ignore")      \
                                            .decode("ascii", "ignore")

                    transcript = transcript.lower().strip()

                    # Convert corresponding FLAC to a WAV
                    flac_file = os.path.join(root, seqid + ".flac")
                    wav_file = os.path.join(target_dir, seqid + ".wav")
                    if not os.path.exists(wav_file):
                        try:
                            Transformer().build(flac_file, wav_file)
                            wav_filesize = os.path.getsize(wav_file)
                            files.append((os.path.abspath(wav_file), wav_filesize, transcript))
                        except OSError:
                            print("Could not find file:", wav_file, flac_file)

    return pandas.DataFrame(data=files, columns=["wav_filename", "wav_filesize", "transcript"])

if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1])
