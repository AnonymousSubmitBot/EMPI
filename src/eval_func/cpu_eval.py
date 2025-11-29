from src.baseline_init.kaes_init import KAESInitializer
from src.baseline_init.obl_init import OBLInitializer
from src.baseline_init.qi_init import QIInitializer
from src.baseline_init.svm_ss import SVMSSInitializer
from src.ea import EA
from src.ea import ea_algs
from src.problem_domain import BaseProblem
from src.types_ import *


def interpolate_solution(base_pop: NpArray, eval_func, gen_num: int = 20):
    pop_size, dim = base_pop.shape[0], base_pop.shape[1]
    fitness = np.array([eval_func(ind) for ind in base_pop])

    elite_size = int(pop_size * 0.1)
    elite_indices = np.argsort(fitness)[::-1][:elite_size]
    elites = base_pop[elite_indices]

    new_pop = [sol for sol in base_pop]
    new_pop_fit = [fit for fit in fitness]
    for _ in range(gen_num):
        parents = np.concatenate((
            elites[np.random.choice(len(elites), size=2, replace=False)],
            base_pop[np.random.choice(len(base_pop), size=2, replace=False)]), axis=0)

        child = np.zeros(dim, dtype=int)
        for d in range(dim):
            if (parents[:, d] == parents[0, d]).all():
                child[d] = parents[0, d]
            else:
                prob = np.mean(parents[:, d])
                child[d] = 1 if np.random.rand() < prob else 0

        new_pop.append(child)
        new_pop_fit.append(eval_func(child))
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


def eval_problem_solutions(problem_path: str = '../../data/problem_instance', solutions: NpArray = None):
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    result = [problem_instance.evaluate(solution) for solution in solutions]
    return np.array(result)


def eval_problem_EA(ins_dir: Union[Path, str] = '../../data/problem_instance', domain: str = "match_max_problem",
                    problem_type: str = "test", dim: int = 30, idx: int = 0, ea_name: str = "elite_ga",
                    ea_args: Dict[str, Any] = None):
    problem_path = Path(ins_dir, problem_type, f"{domain}_{dim}_{idx}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    ea: EA = ea_algs[ea_name](eval_func=problem_instance.evaluate, dim=dim, **ea_args)
    best_x, best_y = ea.run()
    return best_x, best_y


def eval_problem_EA_detail(ins_dir: Union[Path, str] = '../../data/problem_instance', domain: str = "match_max_problem",
                           problem_type: str = "test", dim: int = 30, idx: int = 0, ea_name: str = "elite_ga",
                           ea_args: Dict[str, Any] = None, with_interpolation: bool = False):
    problem_path = Path(ins_dir, problem_type, f"{domain}_{dim}_{idx}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    if with_interpolation and len(ea_args["initial_solutions"]) > 0:
        new_init, _ = interpolate_solution(ea_args["initial_solutions"], problem_instance.evaluate, gen_num=20)
        ea_args["initial_solutions"] = new_init
    ea: EA = ea_algs[ea_name](eval_func=problem_instance.evaluate, dim=dim, **ea_args)
    best_x, best_y = ea.run()
    return best_x, best_y, ea.step_history, ea.quality_table, ea.population_history


def generate_svmss_init_solution(ins_dir: Union[Path, str] = '../../data/problem_instance',
                                 tar_domain: str = "match_max_problem",
                                 tar_type: str = "gate_train", tar_dim: int = 30,
                                 tar_idx: int = 0, max_eval: int = 88):
    problem_path = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    svmss = SVMSSInitializer(pop_size=20, dim=tar_dim, eval_func=problem_instance.evaluate, max_eval=max_eval)
    sols, quals = svmss.initialize()
    return np.array(sols), np.array(quals)


def generate_obl_init_solution(ins_dir: Union[Path, str] = '../../data/problem_instance',
                               tar_domain: str = "match_max_problem",
                               tar_type: str = "gate_train", tar_dim: int = 30,
                               tar_idx: int = 0, max_eval: int = 88):
    problem_path = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    obl = OBLInitializer(pop_size=20, dim=tar_dim, eval_func=problem_instance.evaluate, max_eval=max_eval)
    sols, quals = obl.initialize()
    return np.array(sols), np.array(quals)


def generate_qi_init_solution(ins_dir: Union[Path, str] = '../../data/problem_instance',
                              tar_domain: str = "match_max_problem",
                              tar_type: str = "gate_train", tar_dim: int = 30,
                              tar_idx: int = 0, max_eval: int = 88):
    problem_path = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    obl = QIInitializer(pop_size=112, gen_num=20, dim=tar_dim, eval_func=problem_instance.evaluate, max_eval=132)
    sols, quals = obl.initialize()
    return np.array(sols), np.array(quals)


def generate_kaes_init_solution(ins_dir: Union[Path, str] = '../../data/problem_instance',
                                tar_domain: str = "match_max_problem",
                                tar_type: str = "gate_train", tar_dim: int = 30,
                                tar_idx: int = 0, max_eval: int = 88):
    problem_path = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    kaes = KAESInitializer(pop_size=20, dim=tar_dim, eval_func=problem_instance.evaluate, ins_dir=ins_dir,
                           max_eval=max_eval)
    sols, quals = kaes.initialize()
    return np.array(sols), np.array(quals)


