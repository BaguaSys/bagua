import unittest
import numpy as np
from bagua.service.bayesian_optimizer import (
    BayesianOptimizer,
    IntParam,
)


class TestBayesianOptimizer(unittest.TestCase):
    def test_bayesian_optimization(self):
        def f(x, y):
            x /= 10000.0
            y /= 10000.0
            return np.sin(5 * x) * (1 - np.tanh(y ** 2)) + np.random.randn() * 0.1

        optim = BayesianOptimizer(
            {
                "x": IntParam(
                    val=1,
                    space_dimension=[-20000, 20000],
                ),
                "y": IntParam(
                    val=1,
                    space_dimension=[-20000, 20000],
                ),
            },
        )

        param_score_list = []

        # Tune
        for i in range(100):
            d = optim.ask()
            score = f(d["x"], d["y"])
            optim.tell(d, score)
            param_score_list.append([d, score])

        # Test
        best_score = sorted(param_score_list, key=lambda p_s: -p_s[1])[0][1]
        self.assertTrue(best_score > 0.8)


if __name__ == "__main__":
    unittest.main()
