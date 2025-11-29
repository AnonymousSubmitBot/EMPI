from pymoo.algorithms.soo.nonconvex.brkga import BRKGA as PyMOOBRKGA
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from src.ea.ea import EA
from src.types_ import *


class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return a.get("hash") == b.get("hash")


class MyBitProblem(ElementwiseProblem):

    def __init__(self, eval_func: Callable[[NpArray], float], dimension: int, max_eval: int):
        self.eval_func: Callable[[NpArray], float] = eval_func
        self.max_eval: int = max_eval
        self.eval_time: int = 0
        self.step_history: List = []
        self.quality_table: Dict[str, float] = {}
        super().__init__(n_var=dimension, n_obj=1, n_ieq_constr=0, xl=0, xu=1)

    def out_evaluate(self, x, out):
        self._evaluate(x, out)

    def _evaluate(self, x, out, *args, **kwargs):
        pheno = np.array([0 if value <= 0.5 else 1 for value in x], dtype=np.int32)
        pheno_str = "".join([str(i) for i in pheno])
        if self.eval_time >= self.max_eval:
            eval_value = -1e15
        else:
            if pheno_str in self.quality_table.keys():
                eval_value = self.quality_table[pheno_str]
            else:
                eval_value = self.eval_func(pheno)
                self.quality_table[pheno_str] = eval_value
        self.eval_time += 1
        out["F"] = -eval_value
        out["pheno"] = pheno
        out["hash"] = hash(pheno_str)
        self.step_history.append(eval_value)


class MyBRKGA(PyMOOBRKGA):
    def __init__(self,
                 n_elites=200,
                 n_offsprings=700,
                 n_mutants=100,
                 initial_solutions: List[NpArray] = None,
                 **kwargs
                 ):
        self.pop_history = []
        self.initial_solutions: List[NpArray] = initial_solutions
        super().__init__(n_elites=n_elites, n_offsprings=n_offsprings, n_mutants=n_mutants, **kwargs)

    def run(self):
        while self.has_next():
            self.next()
            self.pop_history.append([-individual.F[0] for individual in self.pop])
        return self.result()

    def _initialize_infill(self):
        if self.initial_solutions is None:
            pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)
        else:
            pop: Population = self.initialization.do(self.problem, self.pop_size - len(self.initial_solutions),
                                                     algorithm=self)
            initial_pop = Population.new("X", self.initial_solutions)
            pop = Population.merge(initial_pop, pop)
        return pop


class BRKGA(EA):
    recommend_kwargs = {
        "max_eval": 800,
        "max_iter": 400,
        "n_elites": 4,
        "n_offsprings": 14,
        "n_mutants": 2,
        "bias": 0.7
    }

    def __init__(self, eval_func: Callable[[NpArray], float], dim: int, max_eval: int = 800, max_iter: int = 120,
                 n_elites: int = 13, n_offsprings: int = 44, n_mutants: int = 7, bias: float = 0.7,
                 initial_solutions: Union[NpArray, List] = None):
        super().__init__(eval_func, dim)
        self.dim = dim
        self.max_eval: int = max_eval
        self.max_iter: int = max_iter
        self.n_elites: int = n_elites
        self.n_offsprings: int = n_offsprings
        self.n_mutants: int = n_mutants
        self.pop_size = self.n_elites + self.n_offsprings + self.n_mutants
        self.bias: float = bias
        self.initial_solutions: Union[NpArray, List] = initial_solutions
        self.problem = MyBitProblem(eval_func=eval_func, dimension=self.dim, max_eval=self.max_eval)
        if len(self.initial_solutions) > self.pop_size:
            [self.problem.out_evaluate(x=solution, out={}) for solution in self.initial_solutions[self.pop_size:]]
            self.initial_solutions = self.initial_solutions[:self.pop_size]
        self.brkga = MyBRKGA(
            n_elites=self.n_elites,
            n_offsprings=self.n_offsprings,
            n_mutants=self.n_mutants,
            bias=self.bias,
            initial_solutions=self.initial_solutions,
            eliminate_duplicates=MyElementwiseDuplicateElimination())
        self.eval_time = 0
        self.quality_table: Dict[str, float] = {}
        self.population_history: List[List[float]] = []
        self.step_history: List[float] = []

    def run(self):
        res = minimize(self.problem,
                       self.brkga,
                       termination=get_termination("n_eval", self.max_eval),
                       verbose=False)
        self.step_history = self.problem.step_history
        self.quality_table = self.problem.quality_table
        self.population_history = res.algorithm.pop_history
        self.eval_time = self.problem.eval_time
        best_x = max(self.quality_table.keys(), key=lambda item: self.quality_table[item])
        best_y = self.quality_table[best_x]
        return best_x, best_y


def test():
    def one_max_fitness(x):
        return np.sum([x[i]*i for i in range(len(x))])

    xs = np.random.randint(0, 2, size=(10, 40))
    xs = np.concatenate((xs,xs), axis=0)
    np.random.seed(1088)
    ga = BRKGA(eval_func=one_max_fitness, dim=40, initial_solutions=np.load(
        "/home/metaron/EMPI_exp/out/svmss_generated_sols/test_contamination_problem_40_1/2/init_0_solutions.npy"))
    ga.run()
    print(ga)


if __name__ == '__main__':
    test()
