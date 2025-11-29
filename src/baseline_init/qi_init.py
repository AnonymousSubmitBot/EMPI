import numpy as np

from src.types_ import *


class QIInitializer:
    def __init__(self, pop_size, gen_num, dim, eval_func, max_eval: int = 132, elite_ratio=0.1):
        self.pop_size = pop_size
        self.gen_num = gen_num
        self.dim = dim
        self.elite_ratio = elite_ratio
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

    def initialize(self):

        base_pop = np.random.randint(0, 2, (self.pop_size, self.dim))

        fitness = np.array([self.get_quality(ind) for ind in base_pop])

        elite_size = int(self.pop_size * self.elite_ratio)
        elite_indices = np.argsort(fitness)[::-1][:elite_size]
        elites = base_pop[elite_indices]

        new_pop = [sol for sol in base_pop]
        new_pop_fit = [fit for fit in fitness]
        for _ in range(self.gen_num):
            parents = np.concatenate((
                elites[np.random.choice(len(elites), size=2, replace=False)],
                base_pop[np.random.choice(len(base_pop), size=2, replace=False)]), axis=0)

            child = np.zeros(self.dim, dtype=int)
            for d in range(self.dim):
                if (parents[:, d] == parents[0, d]).all():
                    child[d] = parents[0, d]
                else:
                    prob = np.mean(parents[:, d])
                    child[d] = 1 if np.random.rand() < prob else 0

            new_pop.append(child)
            new_pop_fit.append(self.get_quality(child))
        sort_idx = np.argsort(new_pop_fit)[::-1]
        new_pop = np.array(new_pop)[sort_idx]
        new_pop_fit = np.array(new_pop_fit)[sort_idx]
        return new_pop, new_pop_fit


def test():
    def sample_fitness(individual):
        return int(''.join(map(str, individual)), 2)


    population_size = 20
    dimension = 8
    init_er = QIInitializer(population_size, dimension, sample_fitness)
    initial_pop, init_qual = init_er.initialize()

    print("Initialized Population:")
    print(initial_pop)
    print("\nCorresponding Fitness Values:")
    for ind in initial_pop:
        print(f"{ind} -> {sample_fitness(ind)}")

if __name__ == '__main__':
    test()