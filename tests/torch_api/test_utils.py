import unittest
import time
import numpy as np
from bagua.torch_api.utils import StatisticalAverage


class TestUtils(unittest.TestCase):
    def test_statistical_average(self):
        m = StatisticalAverage(
            last_update_time=time.time(),
            records=[5.0, 4.5005175, 3.5034241850204078],
            tail=(2.0166309999999985, 2.499061185570463),
        )
        time.sleep(1)

        m.record(6.)

        test_answer_list = [
            (1., 6.),
            (2., (6. + 5.) / 2.),
            (3., (6. + 4.5005175 * 2.) / 3.),
            (5., (6. + 3.5034241850204078 * 4.) / 5.),
            (7., (6. + 2.499061185570463 * (4. + 2.0166309999999985)) / 7.),
        ]
        test_val_list = []
        for (last_n_seconds, _) in test_answer_list:
            test_val_list.append(
                m.get_records_mean(last_n_seconds))

        for a, (n, b) in zip(test_val_list, test_answer_list):
            self.assertTrue(
                np.isclose(a, b, atol=0.1),
                "last_n_seconds={}, test_val={}, answer={}, m={}".format(
                    n, a, b, m
                ))

        self.assertTrue(
            np.isclose(
                m.total_recording_time(),
                4. + 2.0166309999999985 + 1,
                atol=0.1,
            ), "total_recording_time={}".format(m.total_recording_time())
        )


if __name__ == "__main__":
    unittest.main()