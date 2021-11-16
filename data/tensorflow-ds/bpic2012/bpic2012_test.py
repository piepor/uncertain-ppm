"""bpic2012 dataset."""

import tensorflow_datasets as tfds
from . import bpic2012


class Bpic2012Test(tfds.testing.DatasetBuilderTestCase):
  """Tests for bpic2012 dataset."""
  # TODO(bpic2012):
  DATASET_CLASS = bpic2012.Bpic2012
  SPLITS = {
      'train': 2,  # Number of fake train example
      #'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}
  DL_EXTRACT_RESULT = {'file': 'fake_log.xes'}


if __name__ == '__main__':
  tfds.testing.test_main()
