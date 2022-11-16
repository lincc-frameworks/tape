import unittest
from lsstseries import timeseries


class TestReturnValues(unittest.TestCase):
    def test_StetsonJ(self):
        """
        Simple test of StetsonJ function for a known return value
        """
        flux_list = [0, 1, 2, 3, 4]
        test_dict = {'time': range(len(flux_list)), 'flux': flux_list,
                     'flux_err': [1]*len(flux_list), 'band': ['r']*len(flux_list)}
        ts = timeseries()
        test_ts = ts.from_dict(data_dict=test_dict)
        test_ts.stetson_J()['r'] == 0.8
        self.assertEqual(test_ts.stetson_J()['r'],
                         0.8)


if __name__ == '__main__':
    unittest.main()
