from src.types_ import *


class EA:
    recommend_kwargs = {}
    def __init__(self, eval_func: Callable[[NpArray], float], dim: int):
        self.eval_time = 0
        self.quality_table: Dict[str, float] = {}
        self.population_history: List[List[float]] = []
        self.step_history: List[float] = []

    def run(self):
        pass
