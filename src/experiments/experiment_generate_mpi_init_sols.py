import os.path
import random
from multiprocessing import Process, Queue

from src.ea import ea_algs
from src.eval_func.cpu_eval import interpolate_solution
from src.experiments.experiment_eval_init_ea import generate_eval_data
from src.experiments.experiment_problem import load_problem_data, load_sample_indices
from src.network.gate_net import GateNet
from src.problem_domain import train_problem_domains, valid_problem_domains, problem_settings, BaseProblem
from src.types_ import *


def sort_with_high_diversity(sols, quals, elite_num=5, pop_size=20):
    sols, quals = np.array(sols), np.array(quals)
    init_idx = np.argsort(np.array(quals))[::-1]
    sorted_idx, repeat_idx = [], []
    sol_str_set = set()
    for sol_idx in init_idx:
        sol_str = "".join([str(int(i)) for i in sols[sol_idx]])
        if sol_str in sol_str_set:
            repeat_idx.append(sol_idx)
        else:
            sorted_idx.append(sol_idx)
            sol_str_set.add(sol_str)
    sorted_idx, rest_idx = sorted_idx[:elite_num], sorted_idx[elite_num:]
    while len(sorted_idx) < pop_size and len(rest_idx) > 0:
        dis = [np.mean(np.abs([sols[idx] - sols[x] for idx in sorted_idx])) for x in rest_idx]
        dis_rank = np.argsort(np.argsort(dis)[::-1])
        qual_rank = np.array(list(range(len(rest_idx))))
        rank = dis_rank / 2 + qual_rank
        selected = np.argmin(rank)
        sorted_idx.append(rest_idx[selected])
        rest_idx = rest_idx[:selected] + rest_idx[selected + 1:]
    rest_idx = list(rest_idx)
    random.shuffle(rest_idx)
    final_idx = np.array(sorted_idx + rest_idx + repeat_idx)
    return final_idx


def eval_config_on_ins_worker(up_queue: Queue, score_queue: Queue):
    while True:
        data = up_queue.get()
        configs, ins_dir, map_dir, init_batch_size, k, task_idx, init_from_experience, with_interpolation = data
        init_sol_dir = Path(map_dir, "../map_generated_sols")
        (src_type, src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim, tar_idx,
         sample_num, src_coefficient, sample_idx) = configs[0]
        tar_ins_dir = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
        tar_x, tar_y = load_problem_data(problem_dir=tar_ins_dir)
        for idx in range(init_batch_size):
            if init_from_experience:
                store_init_dir = Path(Path(os.path.dirname(os.path.realpath(__file__)), "../../out/mpi_generated_sols"),
                                      f"{tar_type}_{tar_domain}_{tar_dim}_{tar_idx}", f"{sample_idx}",
                                      'interpolation' if with_interpolation else 'no_inter')
                if not os.path.exists(store_init_dir):
                    os.makedirs(store_init_dir)
                sol_path = Path(store_init_dir, f"init_{idx}_solutions.npy")
                qual_path = Path(store_init_dir, f"init_{idx}_qualities.npy")
                if os.path.exists(sol_path) and os.path.exists(qual_path):
                    init_sols = list(np.load(sol_path))
                    init_quals = list(np.load(qual_path))
                else:
                    tar_indices = load_sample_indices(problem_dir=tar_ins_dir, sample_num=sample_num,
                                                      indices_idx=sample_idx)
                    init_sols = list(tar_x[tar_indices])
                    init_quals = list(tar_y[tar_indices])
                    for config in configs:
                        (src_type, src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim, tar_idx,
                         sample_num, src_coefficient, sample_idx) = config
                        map_name = "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
                            "train", src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim,
                            tar_idx, sample_num, src_coefficient, sample_idx
                        )
                        save_dir = Path(init_sol_dir, map_name)
                        sols = list(np.load(Path(save_dir, f"init_{k}_solutions_{idx}.npy")))
                        quals = list(np.load(Path(save_dir, f"init_{k}_qualities_{idx}.npy")))
                        init_sols += sols
                        init_quals += quals
                    init_idx = np.argsort(np.array(init_quals))[::-1]
                    sorted_idx, repeat_idx = [], []
                    sol_str_set = set()
                    for idx in init_idx:
                        sol_str = "".join([str(int(i)) for i in init_sols[idx]])
                        if sol_str in sol_str_set:
                            repeat_idx.append(idx)
                        else:
                            sorted_idx.append(idx)
                            sol_str_set.add(sol_str)
                    init_sols = np.array(init_sols, dtype=np.int64)[np.array(sorted_idx + repeat_idx)]
                    init_quals = np.array(init_quals)[np.array(sorted_idx + repeat_idx)]
                    if with_interpolation:
                        problem_path = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
                        problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
                        init_sols, init_quals = interpolate_solution(init_sols,
                                                                     problem_instance.evaluate, gen_num=20)
                    np.save(sol_path, init_sols)
                    np.save(qual_path, init_quals)
            else:
                store_init_dir = Path(
                    Path(os.path.dirname(os.path.realpath(__file__)), "../../out/random_generated_sols"),
                    f"{tar_type}_{tar_domain}_{tar_dim}_{tar_idx}", f"{sample_idx}",
                    'interpolation' if with_interpolation else 'no_inter')
                if not os.path.exists(store_init_dir):
                    os.makedirs(store_init_dir)
                sol_path = Path(store_init_dir, f"init_{idx}_solutions.npy")
                qual_path = Path(store_init_dir, f"init_{idx}_qualities.npy")
                if os.path.exists(sol_path) and os.path.exists(qual_path):
                    init_sols = list(np.load(sol_path))
                    init_quals = list(np.load(qual_path))
                else:
                    if not os.path.exists(store_init_dir):
                        os.makedirs(store_init_dir)
                    init_sols = np.random.randint(0, 2, size=(20, tar_dim))
                    problem_path = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
                    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
                    init_quals = np.array([problem_instance.evaluate(sol) for sol in init_sols])
                    np.save(sol_path, init_sols)
                    np.save(qual_path, init_quals)
        score_queue.put((task_idx, 1))

def eval_ea(
        ins_dir: Union[Path, str] = "../../data/problem_instance",
        map_dir: Union[Path, str] = "../../out/map_surrogate_models",
        result_dir: Union[Path, str] = "../../out/eval_result",
        weight_dir: Union[Path, str] = "../../out/gate_wights",
        tar_type: str = "test",
        sample_num: int = 64, src_coefficient: int = 4, b: int = 12, k: int = 2,
        sample_batch_size: int = 3, init_batch_size: int = 10,
        source_ea_name: str = "random",
        source_k: int = 0,
        save_result: bool = True,
        with_interpolation: bool = False
):
    data_worker_num, result_worker_num, clean_worker_num = 90, 10, 1
    up_queue, score_queue, result_queue, clean_queue = Queue(), Queue(), Queue(), Queue()
    worker_list: List[Process] = []
    for _ in range(data_worker_num):
        worker = Process(target=eval_config_on_ins_worker, args=(up_queue, score_queue))
        worker.start()
        worker_list.append(worker)
    corr_dict: Dict = pickle.loads(open(Path(corr_dir, f"train-CORR-{tar_type}-64_4.pkl"), "rb").read())
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    all_input, all_config_list = generate_eval_data(
        corr_dict=corr_dict, tar_domains=tar_domains, tar_idx_num=tar_idx_num, sample_batch_size=sample_batch_size,
        src_type="train", tar_type=tar_type, sample_num=sample_num, src_coefficient=src_coefficient
    )
    train_ins_num = len(train_problem_domains) * len(problem_settings["training_dims"]) * tar_idx_num
    if source_ea_name != "random":
        gate_weight = np.load(str(Path(weight_dir, f"{source_ea_name}-{b}-{source_k}", f"train_{source_ea_name}.npy")))
        gate_net = GateNet(feature_num=train_ins_num * 3, candidate_num=train_ins_num, hidden_dims=[128, 128])
        results = gate_net(torch.DoubleTensor(all_input), torch.DoubleTensor(gate_weight))
    else:
        results = [0 for _ in range(len(all_config_list))]
    ea_results = {}
    for ins_idx in range(len(all_config_list)):
        top_k_src = np.argsort(results[ins_idx])[::-1][:b]
        configs = [all_config_list[ins_idx][idx] for idx in top_k_src]
        up_queue.put(
            (configs, ins_dir, map_dir, init_batch_size, k, ins_idx, source_ea_name != "random",
             with_interpolation)
        )
    while len(ea_results) < len(all_config_list):
        task_idx, metric = score_queue.get()
        ea_results[task_idx] = {
            "config": all_config_list[task_idx][0],
            "metric": metric
        }

    [worker.terminate() for worker in worker_list]


if __name__ == '__main__':
    corr_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/correlation_results")
    ins_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    map_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/map_surrogate_models")
    weight_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/gate_weights")
    result_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/eval_result")
    ks = [4]
    source_ea_names = ["no_ea"]
    max_parallel_num = 3
    p_list = []
    for interpolation in [True, False]:
        for k in ks:
            for source_ea_name in source_ea_names:
                print(source_ea_name, k, interpolation)
                eval_ea(ins_dir, map_dir, result_dir, weight_dir, "test", 64,
                                 4, 12, k, 3, 10, source_ea_name, k, True, interpolation)

    eval_ea(ins_dir, map_dir, result_dir, weight_dir, "test", 64,
                     4, 12, 2, 3, 10, "random", 2, True, False)
