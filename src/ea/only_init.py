from src.ea.ea import EA
from src.types_ import *


class OnlyInit(EA):
    recommend_kwargs = {}

    def __init__(self, eval_func: Callable[[NpArray], float], dim: int, initial_solutions: Union[NpArray, List] = None,
                 max_iter: int = 1000, max_eval: int = 8000):
        super().__init__(eval_func, dim)
        self.eval_time = 0
        self.eval_func = eval_func
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.quality_table: Dict[str, float] = {}
        self.population_history: List[List[float]] = []
        self.step_history: List[float] = []
        self.initial_solutions = initial_solutions

    def get_quality(self, solution: NpArray) -> float:
        solution_str = "".join([str(i) for i in solution])
        if self.eval_time >= self.max_eval:
            return -1e15
        self.eval_time += 1
        if solution_str in self.quality_table:
            quality = self.quality_table[solution_str]
        else:
            quality = self.eval_func(solution)
            self.quality_table[solution_str] = quality
        self.step_history.append(quality)
        solution_str = "".join([str(i) for i in solution])
        self.quality_table[solution_str] = quality
        return quality

    def run(self):
        [self.get_quality(solution) for solution in self.initial_solutions]
        best_x = max(self.quality_table.keys(), key=lambda item: self.quality_table[item])
        best_y = self.quality_table[best_x]
        return best_x, best_y
