from src.types_ import *


class OBLInitializer:
    def __init__(self, pop_size, dim, eval_func, max_eval: int = 88):
        self.pop_size = pop_size
        self.dim = dim
        self.eval_func = eval_func
        self.max_eval: int = max_eval
        self.eval_time = 0
        self.quality_table: Dict[str, float] = {}
        self.sol_history: List[Union[NpArray, List]] = []
        self.qual_history: List[float] = []

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
        self.qual_history.append(quality)
        self.sol_history.append(solution)
        solution_str = "".join([str(i) for i in solution])
        self.quality_table[solution_str] = quality
        return quality

    def initialize(self, top_k=None):
        population = np.random.randint(0, 2, size=(self.pop_size, self.dim))

        opposition_population = 1 - population

        combined_population = np.vstack([population, opposition_population])

        [self.get_quality(ind) for ind in combined_population]

        sort_idx = np.argsort(self.qual_history)[::-1]
        self.qual_history = np.array(self.qual_history)[sort_idx]
        self.sol_history = np.array(self.sol_history)[sort_idx]
        if not top_k:
            top_k = len(self.sol_history)
        return self.sol_history[:top_k], self.qual_history[:top_k]


def test():
    def sample_fitness(individual):
        return int(''.join(map(str, individual)), 2)


    population_size = 5
    dimension = 8
    init_er = OBLInitializer(population_size, dimension, sample_fitness)
    initial_pop, init_qual = init_er.initialize()

    print("Initialized Population:")
    print(initial_pop)
    print("\nCorresponding Fitness Values:")
    for ind in initial_pop:
        print(f"{ind} -> {sample_fitness(ind)}")

if __name__ == '__main__':
    test()