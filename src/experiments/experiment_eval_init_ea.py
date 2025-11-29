import os.path
import random
import zipfile
from io import BytesIO
from multiprocessing import Process, Queue

import requests

from src.distribution.util import random_str
from src.ea import ea_algs
from src.experiments.experiment_problem import load_problem_data, load_sample_indices
from src.experiments_setting import master_host
from src.network.gate_net import GateNet
from src.problem_domain import train_problem_domains, valid_problem_domains, problem_cost, problem_settings
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


def generate_eval_data(corr_dict, tar_domains, tar_idx_num, sample_batch_size, src_type, tar_type, sample_num,
                       src_coefficient) -> Tuple[NpArray, List]:
    all_input = []
    all_config_list = []
    for tar_domain in tar_domains.keys():
        for tar_dim in problem_settings["valid_dims"]:
            for tar_idx in range(tar_idx_num):
                for sample_idx in range(sample_batch_size):
                    input, config_list = [], []
                    for src_domain in train_problem_domains.keys():
                        for src_dim in problem_settings["training_dims"]:
                            for src_idx in range(problem_settings["training_ins_num"]):
                                map_name = "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
                                    src_type, src_domain, src_dim, src_idx,
                                    tar_type, tar_domain, tar_dim, tar_idx,
                                    sample_num, src_coefficient, sample_idx
                                )
                                config_list.append(
                                    (src_type, src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim, tar_idx,
                                     sample_num, src_coefficient, sample_idx)
                                )
                                input += [corr_dict[map_name]['pearson'], corr_dict[map_name]['spearman'],
                                          corr_dict[map_name]['kendall']]
                    all_input.append(input)
                    all_config_list.append(config_list)
    return np.array(all_input, dtype=np.float64), all_config_list


def send_ea_eval_tasks(all_task_args, min_y, max_y, task_idx, result_queue: Queue):
    all_task_id = []
    for task_args in all_task_args:
        task_id = f"{str(int(time.time() * 1000))}_{random_str(32)}"
        all_task_id.append(task_id)
        task = {
            "task_id": task_id,
            "task_func": "eval_problem_EA_detail",
            "task_args": task_args,
            "task_cost": problem_cost[task_args[1]],
            "task_type": "cpu"
        }
        task_data = pickle.dumps(task)
        response_code = 500
        while response_code > 299 or response_code < 200:
            try:
                response = requests.post(url="http://{}/create_task".format(master_host),
                                         data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                               "task_type": "cpu", "task_cost": problem_cost[task_args[1]],
                                               "task_id": task_id},
                                         files={"task_data": task_data},
                                         verify=False)
                response_code = response.status_code
            except Exception:
                pass
    result_queue.put((all_task_id, min_y, max_y, task_idx))


def send_get_result_worker(result_queue: Queue, score_queue: Queue, clean_queue: Queue):
    while True:
        data = result_queue.get()
        all_task_id, min_y, max_y, task_idx = data
        try:
            response = requests.post(url="http://{}/get_multi_result".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_id": all_task_id},
                                     verify=False)
            response_code = response.status_code
        except Exception:
            result_queue.put(data)
            continue
        if response_code > 299 or response_code < 200:
            result_queue.put(data)
        else:
            score_dict = {}
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir(): continue
                    with zip_ref.open(file_info) as file:
                        file_data = file.read()
                        result = pickle.loads(file_data)
                        score_dict[file_info.filename] = {
                            "best_x": result[0],
                            "best_y": result[1],
                            "step_history": result[2],
                            "pop_history": result[4]
                        }
            if len(score_dict) != len(all_task_id):
                print("GET RESULT ERROR!!!")
                result_queue.put(data)
            else:
                all_scores = [score_dict[task_id] for task_id in all_task_id]
                score_queue.put((task_idx, all_scores))
                for task_id in all_task_id:
                    clean_queue.put(task_id)


def clean_record_worker(clean_queue: Queue):
    while True:
        task_id = clean_queue.get()
        try:
            response = requests.post(url="http://{}/clear_record".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_id": task_id},
                                     verify=False)
            response_code = response.status_code
        except Exception:
            clean_queue.put(task_id)
            continue
        if response_code > 299 or response_code < 200:
            clean_queue.put(task_id)


def eval_config_on_ins_worker(up_queue: Queue, result_queue: Queue):
    while True:
        data = up_queue.get()
        configs, ins_dir, map_dir, init_batch_size, k, ea_name, task_idx, init_from_experience, with_interpolation = data
        init_sol_dir = Path(map_dir, "../map_generated_sols")
        all_task_args = []
        (src_type, src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim, tar_idx,
         sample_num, src_coefficient, sample_idx) = configs[0]
        tar_ins_dir = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
        tar_x, tar_y = load_problem_data(problem_dir=tar_ins_dir)
        ea = ea_algs[ea_name]
        ea_args = ea.recommend_kwargs.copy()
        for idx in range(init_batch_size):
            if init_from_experience:
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
            else:
                init_sols = np.array([])
            ea_args["initial_solutions"] = np.array(init_sols)
            ea_args["max_eval"] = 1600
            ea_args["max_iter"] = 4000
            task_args = (ins_dir, tar_domain, tar_type, tar_dim, tar_idx, ea_name, ea_args, with_interpolation)
            all_task_args.append(task_args)
        send_ea_eval_tasks(all_task_args, np.min(tar_y), np.max(tar_y), task_idx, result_queue)


def eval_ea(
        ins_dir: Union[Path, str] = "../../data/problem_instance",
        map_dir: Union[Path, str] = "../../out/map_surrogate_models",
        result_dir: Union[Path, str] = "../../out/eval_result",
        weight_dir: Union[Path, str] = "../../out/gate_wights",
        tar_type: str = "test", ea_name: str = "elite_ga",
        sample_num: int = 64, src_coefficient: int = 4, b: int = 12, k: int = 2,
        sample_batch_size: int = 3, init_batch_size: int = 10,
        source_ea_name: str = "random",
        source_k: int = 0,
        save_result: bool = True,
        with_interpolation: bool = False,
        random_src: bool = False,
):
    init_name = (f"{source_ea_name}{'-interpolation' if with_interpolation else ''}"
                 f"{'-random_select' if random_src else ''}-{b}-{source_k}")
    if save_result:
        result_name = f"{tar_type}-{ea_name}-{f'{init_name}-{k}' if source_ea_name != 'random' else 'random'}.pkl"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if os.path.exists(Path(result_dir, result_name)):
            print(f"{result_name} EXIST")
            return
    data_worker_num, result_worker_num, clean_worker_num = 90, 10, 1
    up_queue, score_queue, result_queue, clean_queue = Queue(), Queue(), Queue(), Queue()
    worker_list: List[Process] = []
    for _ in range(data_worker_num):
        worker = Process(target=eval_config_on_ins_worker, args=(up_queue, result_queue))
        worker.start()
        worker_list.append(worker)
    for _ in range(result_worker_num):
        worker = Process(target=send_get_result_worker, args=(result_queue, score_queue, clean_queue))
        worker.start()
        worker_list.append(worker)
    for _ in range(clean_worker_num):
        worker = Process(target=clean_record_worker, args=(clean_queue,))
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
    if source_ea_name != "random" and random_src is False:
        gate_weight = np.load(str(Path(weight_dir, f"{source_ea_name}-{b}-{source_k}", f"train_{source_ea_name}.npy")))
        gate_net = GateNet(feature_num=train_ins_num * 3, candidate_num=train_ins_num, hidden_dims=[128, 128])
        results = gate_net(torch.DoubleTensor(all_input), torch.DoubleTensor(gate_weight))
    else:
        results = [0 for _ in range(len(all_config_list))]
    ea_results = {}
    for ins_idx in range(len(all_config_list)):
        if random_src:
            top_k_src = random.sample(range(len(all_config_list[ins_idx])), b)
        else:
            top_k_src = np.argsort(results[ins_idx])[::-1][:b]
        configs = [all_config_list[ins_idx][idx] for idx in top_k_src]
        up_queue.put(
            (configs, ins_dir, map_dir, init_batch_size, k, ea_name, ins_idx, source_ea_name != "random",
             with_interpolation)
        )
    while len(ea_results) < len(all_config_list):
        task_idx, metric = score_queue.get()
        ea_results[task_idx] = {
            "config": all_config_list[task_idx][0],
            "metric": metric
        }
    if save_result:
        result_name = f"{tar_type}-{ea_name}-{f'{init_name}-{k}' if source_ea_name != 'random' else 'random'}.pkl"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        open(Path(result_dir, result_name), "wb").write(pickle.dumps(ea_results))

    [worker.terminate() for worker in worker_list]


if __name__ == '__main__':
    corr_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/correlation_results")
    ins_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    map_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/map_surrogate_models")
    weight_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/gate_weights")
    result_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/eval_result")
    ks = [4]
    source_ea_names = ["no_ea", "no_ea_diversity", "no_ea_max", "no_ea_max_diversity"]
    ea_names = ["brkga", "elite_ga", "only_init"]
    max_parallel_num = 3
    p_list = []
    for ea_name in ea_names:
        for random_select in [False]:
            for interpolation in [True, False]:
                for k in ks:
                    for source_ea_name in source_ea_names:
                        while len(p_list) >= max_parallel_num:
                            for process in p_list:
                                if not process.is_alive():
                                    process.join()
                                    process.close()
                                    p_list.remove(process)
                        print(source_ea_name, k, ea_name, interpolation, random_select)
                        p = Process(target=eval_ea,
                                    args=(ins_dir, map_dir, result_dir, weight_dir, "test", ea_name, 64,
                                          4, 12, k, 3, 10, source_ea_name, k, True, interpolation, random_select))
                        p.start()
                        p_list.append(p)
        if ea_name != "only_init":
            while len(p_list) >= max_parallel_num:
                for process in p_list:
                    if not process.is_alive():
                        process.join()
                        process.close()
                        p_list.remove(process)
            print(ea_name, "random")
            p = Process(target=eval_ea,
                        args=(ins_dir, map_dir, result_dir, weight_dir, "test", ea_name, 64,
                              4, 12, 2, 3, 10, "random", 2, True, False, False))
            p.start()
            p_list.append(p)
    [p.join() for p in p_list]
