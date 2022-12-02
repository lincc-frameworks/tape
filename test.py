import unittest
from lsstseries import ensemble, timeseries


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
        print('test StetsonJ value is: ' + str(test_ts.stetson_J()['r']))
        self.assertEqual(test_ts.stetson_J()['r'],
                         0.8)

    def test_build_index(self):
        """
        Test that ensemble indexing returns expected behavior
        """
        obj_ids = [1, 1, 1, 2, 1, 2, 2]
        bands = ['u', 'u', 'u', 'g', 'g', 'u', 'u']

        ens = ensemble()
        result = list(ens._build_index(obj_ids, bands).get_level_values(2))
        target = [0, 1, 2, 0, 0, 0, 1]
        self.assertEqual(result, target)


if __name__ == '__main__':
    unittest.main()
