from scipy.special import gamma
from scipy.stats import qmc
from sklearn.svm import SVC

from src.types_ import *


class SVMSSInitializer:
    def __init__(self, pop_size, dim, eval_func, K=50, levy_beta=1.5, max_iter=100, golden_iter=100,
                 max_eval: int = 88):
        self.random_init_num = pop_size
        self.dim = dim
        self.K = K
        self.levy_beta = levy_beta
        self.max_iter = max_iter
        self.golden_iter = golden_iter
        self.eval_func = eval_func
        self.max_eval: int = max_eval
        self.eval_time = 0
        self.quality_table: Dict[str, float] = {}
        self.sol_history: List[Union[NpArray, List]] = []
        self.qual_history: List[float] = []

    def get_quality(self, solution: NpArray) -> float:
        solution_str = "".join([str(i) for i in solution])
        if self.eval_time >= self.max_eval:
            return 1e15
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
        return -quality

    def initialize(self, top_k=None):
        population_cont = self.lhs_sample(self.random_init_num, self.dim)
        population_bin = (population_cont > 0.5).astype(int)
        fitness = np.array([self.get_quality(ind) for ind in population_bin])

        for _ in range(self.max_iter):
            if self.eval_time >= self.max_eval:
                break
            sorted_indices = np.argsort(fitness)
            labels = np.zeros(self.random_init_num)
            half = self.random_init_num // 2
            labels[sorted_indices[:half]] = 1
            labels[sorted_indices[half:]] = -1

            clf = SVC(kernel='rbf', gamma='scale')
            clf.fit(population_bin, labels)

            top_k_indices = sorted_indices[:self.K]
            weights = np.arange(self.K, 0, -1)
            weights_sum = np.sum(weights)
            centroid = np.zeros(self.dim)
            for i, idx in enumerate(top_k_indices):
                centroid += population_cont[idx] * weights[i]
            centroid /= weights_sum

            levy_step = self.levy_flight(self.dim)
            x_new_cont = centroid + levy_step
            x_new_cont = np.clip(x_new_cont, 0, 1)
            x_new_bin = (x_new_cont > 0.5).astype(int)
            new_fitness = self.get_quality(x_new_bin)
            if self.eval_time >= self.max_eval:
                break

            worst_idx = sorted_indices[-1]
            if new_fitness < fitness[worst_idx]:
                l = np.random.randint(0, self.random_init_num)
                a, b = x_new_cont.copy(), population_cont[l].copy()
                for _ in range(self.golden_iter):
                    c = self.golden_point(a, b, 0.618)
                    d = self.golden_point(b, a, 0.618)
                    c_bin = (c > 0.5).astype(int)
                    d_bin = (d > 0.5).astype(int)

                    c_label = clf.predict([c_bin])[0]
                    d_label = clf.predict([d_bin])[0]

                    if c_label == 1 and d_label == -1:
                        b = d.copy()
                    elif c_label == -1 and d_label == 1:
                        a = c.copy()
                    else:
                        if np.random.rand() < 0.5:
                            a = c.copy()
                        else:
                            b = d.copy()

                final_cont = a if clf.predict([(a > 0.5).astype(int)])[0] == 1 else b
                final_bin = (final_cont > 0.5).astype(int)
                final_fitness = self.get_quality(final_bin)
                if self.eval_time >= self.max_eval:
                    break

                population_cont[worst_idx] = final_cont
                population_bin[worst_idx] = final_bin
                fitness[worst_idx] = final_fitness

        sort_idx = np.argsort(self.qual_history)[::-1]
        self.qual_history = np.array(self.qual_history)[sort_idx]
        self.sol_history = np.array(self.sol_history)[sort_idx]
        if not top_k:
            top_k = len(self.sol_history)
        return self.sol_history[:top_k], self.qual_history[:top_k]

    def lhs_sample(self, n, dim):
        sampler = qmc.LatinHypercube(d=dim)
        sample = sampler.random(n)
        return sample

    def levy_flight(self, size):
        beta = self.levy_beta
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / (np.abs(v) ** (1 / beta))

    def golden_point(self, a, b, rho):
        return rho * a + (1 - rho) * b
