import random
from multiprocessing import Manager, Process
from threading import Thread

import requests
import urllib3

from src.distribution.util import random_str
from src.experiments_setting import master_host
from src.problem_domain import train_problem_domains, valid_problem_domains, problem_cost, problem_settings
from src.types_ import *

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)


def load_sample_indices(problem_dir: Path, sample_num: int = 64, indices_idx: int = 0) -> NpArray:
    sample_dir = Path(problem_dir, "indices")
    if not sample_dir.exists():
        os.makedirs(sample_dir)
    indices_path = Path(sample_dir, f"indices_{indices_idx}.npy")
    if os.path.exists(indices_path):
        return np.load(open(indices_path, 'rb'))[:sample_num]
    else:
        x, y = load_problem_data(problem_dir=problem_dir)
        length = len(x)
        indices = np.arange(length)
        np.random.shuffle(indices)
        np.save(open(indices_path, "wb"), indices)
        return indices[:sample_num]


def send_problem_eval_task(domain, problem_path: Union[Path, str], solutions: NpArray, solution_index: int,
                           result_dict: dict):
    task_id = f"{str(int(time.time() * 1000))}_{random_str(32)}"
    task = {
        "task_id": task_id,
        "task_func": "eval_problem_solutions",
        "task_args": (problem_path, solutions),
        "task_cost": problem_cost[domain],
        "task_type": "cpu"
    }
    task_data = pickle.dumps(task)
    response_code = 500
    while response_code > 299 or response_code < 200:
        try:
            response = requests.post(url="http://{}/create_task".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_type": "cpu", "task_cost": problem_cost[domain], "task_id": task_id},
                                     files={"task_data": task_data},
                                     verify=False)
            response_code = response.status_code
        except Exception:
            pass
    response_code = 500
    while response_code > 299 or response_code < 200:
        try:
            time.sleep(1)
            response = requests.post(url="http://{}/get_result".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_id": task_id},
                                     verify=False)
            response_code = response.status_code
        except Exception:
            pass
    result = pickle.loads(response.content)
    result_dict[solution_index] = result
    while response_code > 299 or response_code < 200:
        try:
            time.sleep(1)
            response = requests.post(url="http://{}/clear_record".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_id": task_id},
                                     verify=False)
            response_code = response.status_code
        except Exception:
            pass


def load_problem_data(problem_dir):
    x = np.load(str(Path(problem_dir, "x.npy")))
    y = np.load(str(Path(problem_dir, "y.npy")))
    return x, y


def generate_instance(problem_class, dim, path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("mkdir -p ", str(path))
        if not os.path.exists(Path(path, "problem.pkl")):
            problem_instance = problem_class(dimension=dim)
            pickle.dump(problem_instance, open(Path(path, "problem.pkl"), "wb"))


def generate_problem_solution(domain: str, problem_path: Union[Path, str], dim: int, sample_num: int):
    if not (os.path.exists(Path(problem_path, "x.npy")) and os.path.exists(Path(problem_path, "y.npy"))):
        solutions = set()
        for _ in range(sample_num):
            temp = tuple(random.randint(0, 1) for _ in range(dim))
            while temp in solutions:
                temp = tuple(random.randint(0, 1) for _ in range(dim))
            solutions.add(temp)
        X: NpArray = np.array(list(solutions), dtype=np.float32)
        x_batches = np.array_split(X, 100)
        manager = Manager()
        y_batch_dict = manager.dict()
        p_list = []
        for batch_index, x_batch in enumerate(x_batches):
            p = Thread(target=send_problem_eval_task, args=(domain, problem_path, x_batch, batch_index, y_batch_dict))
            p_list.append(p)
            p.start()
            time.sleep(0.1)
        [p.join() for p in p_list]
        x = np.concatenate([x_batches[batch_index] for batch_index in range(len(x_batches))])
        y = np.concatenate([y_batch_dict[batch_index] for batch_index in range(len(x_batches))])
        np.save(str(Path(problem_path, "x.npy")), x)
        np.save(str(Path(problem_path, "y.npy")), y)
        for idx in range(10):
            load_sample_indices(problem_path, 16, idx)
        print("Generate New DATA for", str(problem_path))
    if not (os.path.exists(Path(problem_path, "x_inverse.npy")) and os.path.exists(
            Path(problem_path, "y_inverse.npy"))):
        X_ori: NpArray = np.load(str(Path(problem_path, "x.npy")))
        X_inv: NpArray = 1 - X_ori
        x_batches = np.array_split(X_inv, 100)
        manager = Manager()
        y_batch_dict = manager.dict()
        p_list = []
        for batch_index, x_batch in enumerate(x_batches):
            p = Thread(target=send_problem_eval_task, args=(domain, problem_path, x_batch, batch_index, y_batch_dict))
            p_list.append(p)
            p.start()
            time.sleep(0.1)
        [p.join() for p in p_list]
        x = np.concatenate([x_batches[batch_index] for batch_index in range(len(x_batches))])
        y = np.concatenate([y_batch_dict[batch_index] for batch_index in range(len(x_batches))])
        np.save(str(Path(problem_path, "x_inverse.npy")), x)
        np.save(str(Path(problem_path, "y_inverse.npy")), y)
        for idx in range(10):
            load_sample_indices(problem_path, 16, idx)
        print("Generate Inverse DATA for", str(problem_path))


def generate_all_problem_instance(ins_dir: Union[Path, str] = "../../data/problem_instance"):
    if not os.path.exists(Path(ins_dir, "train")):
        os.makedirs(Path(ins_dir, "train"))
    if not os.path.exists(Path(ins_dir, "test")):
        os.makedirs(Path(ins_dir, "test"))
    if not os.path.exists(Path(ins_dir, "gate_train")):
        os.makedirs(Path(ins_dir, "gate_train"))
    for problem_domain in train_problem_domains.keys():
        for index in range(problem_settings["training_ins_num"]):
            for dim in problem_settings["training_dims"]:
                path = Path(ins_dir, "train", f"{problem_domain}_{dim}_{index}")
                generate_instance(train_problem_domains[problem_domain], dim, path)
            for dim in problem_settings["valid_dims"]:
                path = Path(ins_dir, "gate_train", f"{problem_domain}_{dim}_{index}")
                generate_instance(train_problem_domains[problem_domain], dim, path)
    for problem_domain in valid_problem_domains.keys():
        for index in range(problem_settings["valid_ins_num"]):
            for dim in problem_settings["valid_dims"]:
                path = Path(ins_dir, "test", f"{problem_domain}_{dim}_{index}")
                generate_instance(valid_problem_domains[problem_domain], dim, path)


def generate_all_problem_solution_data(ins_dir: Union[Path, str] = "../../data/problem_instance"):
    p_list = []
    for problem_domain in train_problem_domains.keys():
        for index in range(problem_settings["training_ins_num"]):
            for dim in problem_settings["training_dims"]:
                path = Path(ins_dir, "train", f"{problem_domain}_{dim}_{index}")
                p = Process(target=generate_problem_solution, args=(problem_domain, path, dim, 100000))
                p_list.append(p)
                p.start()
            for dim in problem_settings["valid_dims"]:
                path = Path(ins_dir, "gate_train", f"{problem_domain}_{dim}_{index}")
                p = Process(target=generate_problem_solution, args=(problem_domain, path, dim, 100000))
                p_list.append(p)
                p.start()
    for problem_domain in valid_problem_domains.keys():
        for index in range(problem_settings["valid_ins_num"]):
            for dim in problem_settings["valid_dims"]:
                path = Path(ins_dir, "test", f"{problem_domain}_{dim}_{index}")
                p = Process(target=generate_problem_solution, args=(problem_domain, path, dim, 100000))
                p_list.append(p)
                p.start()
    print("WAIT EVAL")
    [p.join() for p in p_list]


if __name__ == "__main__":
    work_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    generate_all_problem_instance(ins_dir=work_dir)
    generate_all_problem_solution_data(work_dir)
