import collections
import logging
import skopt
from typing import Tuple, Optional


class IntParam:
    def __init__(self, val: int, space_dimension: Tuple[int, int]):
        self.val: int = val
        self.space_dimension: Tuple[int, int] = space_dimension

    def __str__(self):
        return str(self.__dict__)


class FloatParam:
    def __init__(self, val: float, space_dimension: Tuple[float, float]):
        self.val: float = val
        self.space_dimension: Tuple[float, float] = space_dimension

    def __str__(self):
        return str(self.__dict__)


class BoolParam:
    def __init__(self, val: bool):
        self.val: bool = val
        self.space_dimension: Tuple[int, int] = (0, 1)

    def __str__(self):
        return str(self.__dict__)


class BayesianOptimizer:
    """
    Simple package of beyasian optimizer
    """

    def __init__(
        self,
        param_declaration: dict,
        n_initial_points: int = 20,
        initial_point_generator: str = "halton",
        random_state: Optional[int] = 0,
    ):
        self.param_declaration = collections.OrderedDict(param_declaration)
        search_space = [
            declar.space_dimension for _, declar in self.param_declaration.items()
        ]

        self.bayesian_optimizer = skopt.Optimizer(
            dimensions=search_space,
            n_initial_points=n_initial_points,
            initial_point_generator=initial_point_generator,
            n_jobs=-1,
            random_state=random_state,
        )

    def tell(self, param_dict: dict, score: float) -> None:
        param_v = [
            float(param_dict[name])
            for name, _ in self.param_declaration.items()  # noqa: E501
        ]
        try:
            self.bayesian_optimizer.tell(param_v, -score)
        except ValueError as err:
            logging.warning(
                "Maybe sklearn's division by 0 bug, skip it. err={}, param_v={}".format(
                    err, param_v
                )
            )

    def ask(self) -> dict:
        param_v = self.bayesian_optimizer.ask()
        param_dict = {}
        for i, (name, _) in enumerate(self.param_declaration.items()):
            param_dict[name] = param_v[i]

        return param_dict
