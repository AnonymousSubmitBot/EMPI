import random

from src.experiments.experiment_problem import load_sample_indices, load_problem_data
from src.problem_domain import train_problem_domains, problem_settings
from src.types_ import *


def calculate_KLD(P, Q):
    max_dim = max(P.shape[1], Q.shape[1])
    P_pad = np.pad(P, [(0, 0), (0, max_dim - P.shape[1])])
    Q_pad = np.pad(Q, [(0, 0), (0, max_dim - Q.shape[1])])

    mu_p = np.mean(P_pad, axis=0)
    mu_q = np.mean(Q_pad, axis=0)
    cov_p = np.cov(P_pad, rowvar=False)
    cov_q = np.cov(Q_pad, rowvar=False)

    cov_p += np.eye(cov_p.shape[0]) * 1e-6
    cov_q += np.eye(cov_q.shape[0]) * 1e-6

    d = len(mu_p)
    cov_q_inv = np.linalg.inv(cov_q)
    term1 = np.trace(cov_q_inv @ cov_p)
    term2 = (mu_p - mu_q) @ cov_q_inv @ (mu_p - mu_q)
    term3 = np.log(np.linalg.det(cov_q) / np.linalg.det(cov_p))
    kld = 0.5 * (term1 + term2 - d + term3)

    return max(kld, 0)


def linear_mapping(P, Q):
    P_inv = np.linalg.pinv(P.T @ P)
    M = Q.T @ P @ P_inv
    return M


def polynomial_kernel(X, Y, degree=5, coef0=0.1):
    """
    κ(x,y) = (x^T y + coef0)^degree
    """
    return (X @ Y.T + coef0) ** degree


def kernel_mapping(P, Q, kernel_func=None):
    if kernel_func is None:
        kernel_func = lambda X, Y: polynomial_kernel(X, Y)

    K = kernel_func(P, P)
    K_inv = np.linalg.pinv(K @ K.T)
    M_k = Q.T @ K.T @ K_inv

    return M_k


def transfer_solutions(history_solutions, mapping, mapping_type='linear', P=None, kernel_func=None):
    if mapping_type == 'linear':
        transferred = history_solutions @ mapping.T
    else:
        K_ps = kernel_func(P, history_solutions)
        transferred = mapping @ K_ps
        transferred = transferred.T

    return np.where(transferred >= 0.5, 1, 0)


def adaptive_mapping_selection(P, Q):
    kld = calculate_KLD(P, Q)
    d = min(P.shape[1], Q.shape[1])
    prob_linear = max(0, 1 - np.sqrt(kld / (2 * d)))
    return prob_linear


class KAESInitializer:
    def __init__(self, pop_size, dim, eval_func, ins_dir, max_eval=132):
        self.pop_size = pop_size
        self.dim = dim
        self.eval_func = eval_func
        self.ins_dir = ins_dir
        self.max_eval = max_eval
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

    def generate_from_one_exp(self, selected_exp, P_curr, sample_num, gen_num):
        src_ins_dir = Path(self.ins_dir, "train", selected_exp)
        src_indices = load_sample_indices(src_ins_dir, sample_num=sample_num,
                                          indices_idx=random.randint(0, 9))
        src_x, src_y = load_problem_data(problem_dir=src_ins_dir)
        P_hist, Q_hist = src_x[src_indices], src_y[src_indices]
        sort_indices = np.argsort(Q_hist)[::-1]
        P_hist, Q_hist = P_hist[sort_indices], Q_hist[sort_indices]
        prob_linear = adaptive_mapping_selection(P_hist, P_curr)
        use_linear = np.random.rand() < prob_linear
        top_hist = src_x[np.argsort(src_y)[::-1][:gen_num]]
        if use_linear:
            M = linear_mapping(P_hist, P_curr)
            transferred_sols = transfer_solutions(top_hist, M, 'linear')
        else:
            M_k = kernel_mapping(P_hist, P_curr)
            transferred_sols = transfer_solutions(
                top_hist,
                M_k,
                'kernel',
                P=P_hist,
                kernel_func=polynomial_kernel
            )
        return transferred_sols

    def initialize(self):
        all_train_ins = []
        for src_domain in train_problem_domains.keys():
            for src_dim in problem_settings["training_dims"]:
                for src_idx in range(problem_settings["training_ins_num"]):
                    all_train_ins.append(f"{src_domain}_{src_dim}_{src_idx}")
        gen_num = self.pop_size // len(all_train_ins) + 1
        sample_num = self.max_eval - gen_num * len(all_train_ins)
        P_curr = np.random.randint(0, 2, size=(sample_num, self.dim))
        Q_curr = np.array([self.get_quality(solution) for solution in P_curr])
        sort_indices = np.argsort(Q_curr)[::-1]
        P_curr, Q_curr = P_curr[sort_indices], Q_curr[sort_indices]
        new_pop = [sol for sol in P_curr]
        new_pop_fit = [fit for fit in Q_curr]
        for exp in all_train_ins:
            transferred_sols = self.generate_from_one_exp(exp, P_curr, sample_num, gen_num)
            transferred_quals = [self.get_quality(solution) for solution in transferred_sols]
            [new_pop.append(solution) for solution in transferred_sols]
            [new_pop_fit.append(fit) for fit in transferred_quals]
        init_idx = np.argsort(new_pop_fit)[::-1]
        sorted_idx, repeat_idx = [], []
        sol_str_set = set()
        for idx in init_idx:
            sol_str = "".join([str(int(i)) for i in new_pop[idx]])
            if sol_str in sol_str_set:
                repeat_idx.append(idx)
            else:
                sorted_idx.append(idx)
                sol_str_set.add(sol_str)
        new_pop = np.array(new_pop, dtype=np.int64)[np.array(sorted_idx + repeat_idx)]
        new_pop_fit = np.array(new_pop_fit)[np.array(sorted_idx + repeat_idx)]
        return new_pop, new_pop_fit


def test():
    ins_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    problem_path = "/home/metaron/EMPI_exp/data/problem_instance/test/contamination_problem_80_2"
    problem_instance = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    kaes = KAESInitializer(pop_size=20, dim=80, eval_func=problem_instance.evaluate, ins_dir=ins_dir, max_eval=132)
    a, b = kaes.initialize()
    print(len(b), b)


if __name__ == '__main__':
    test()
