"""helpdesk dataset."""

import tensorflow_datasets as tfds
import helpdesk


class HelpdeskTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for helpdesk dataset."""
  # TODO(helpdesk):
  DATASET_CLASS = helpdesk.Helpdesk
  SPLITS = {
      'train': 2,  # Number of fake train example
      #'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}
  DL_EXTRACT_RESULT = {'file': 'finale.csv'}


if __name__ == '__main__':
  tfds.testing.test_main()
