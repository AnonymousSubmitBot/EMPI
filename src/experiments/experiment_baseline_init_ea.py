import os.path
from multiprocessing import Process, Queue

from src.ea import ea_algs
from src.eval_func import generate_svmss_init_solution, generate_obl_init_solution, generate_qi_init_solution
from src.eval_func.cpu_eval import generate_kaes_init_solution
from src.experiments.experiment_eval_init_ea import send_ea_eval_tasks, send_get_result_worker
from src.experiments.experiment_problem import load_problem_data
from src.experiments.experiment_train_gate import clean_record_worker
from src.problem_domain import train_problem_domains, problem_settings, valid_problem_domains
from src.types_ import *

baseline_evals = {
    "SVMSS": generate_svmss_init_solution,
    "OBL": generate_obl_init_solution,
    "QI": generate_qi_init_solution,
    "KAES": generate_kaes_init_solution
}

baseline_init_dirs = {
    "SVMSS": Path(os.path.dirname(os.path.realpath(__file__)), "../../out/svmss_generated_sols"),
    "OBL": Path(os.path.dirname(os.path.realpath(__file__)), "../../out/obl_generated_sols"),
    "QI": Path(os.path.dirname(os.path.realpath(__file__)), "../../out/qi_generated_sols"),
    "KAES": Path(os.path.dirname(os.path.realpath(__file__)), "../../out/kaes_generated_sols"),
}


def eval_config_on_ins_worker(up_queue: Queue, result_queue: Queue):
    while True:
        data = up_queue.get()
        config, ins_dir, save_init_dir, init_batch_size, init_eval_times, ea_name, baseline, task_idx = data
        all_task_args = []
        (tar_type, tar_domain, tar_dim, tar_idx, sample_idx) = config
        tar_ins_dir = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
        tar_x, tar_y = load_problem_data(problem_dir=tar_ins_dir)
        ea = ea_algs[ea_name]
        ea_args = ea.recommend_kwargs.copy()
        store_init_dir = Path(save_init_dir, f"{tar_type}_{tar_domain}_{tar_dim}_{tar_idx}", f"{sample_idx}",
                              f"{init_eval_times}")
        if not os.path.exists(store_init_dir):
            os.makedirs(store_init_dir)
        for idx in range(init_batch_size):
            (tar_type, tar_domain, tar_dim, tar_idx, sample_idx) = config
            sol_path = Path(store_init_dir, f"init_{idx}_solutions.npy")
            qual_path = Path(store_init_dir, f"init_{idx}_qualities.npy")
            if os.path.exists(sol_path) and os.path.exists(qual_path):
                init_sols = list(np.load(sol_path))
                init_quals = list(np.load(qual_path))
            else:
                sols, quals = baseline_evals[baseline](ins_dir=ins_dir, tar_domain=tar_domain, tar_type=tar_type,
                                                       tar_dim=tar_dim, tar_idx=tar_idx, max_eval=init_eval_times)
                np.save(sol_path, sols)
                np.save(qual_path, quals)
                init_sols = sols
                init_quals = quals
            init_idx = np.argsort(np.array(init_quals))[::-1]
            init_sols = np.array(init_sols, dtype=np.int64)[init_idx]
            ea_args["initial_solutions"] = np.array(init_sols)
            ea_args["max_eval"] = 1600
            ea_args["max_iter"] = 4000
            task_args = (ins_dir, tar_domain, tar_type, tar_dim, tar_idx, ea_name, ea_args, False)
            all_task_args.append(task_args)
        send_ea_eval_tasks(all_task_args, np.min(tar_y), np.max(tar_y), task_idx, result_queue)


def generate_eval_data(tar_domains, tar_idx_num, sample_batch_size, tar_type) -> List:
    all_config_list = []
    for tar_domain in tar_domains.keys():
        for tar_dim in problem_settings["valid_dims"]:
            for tar_idx in range(tar_idx_num):
                for sample_idx in range(sample_batch_size):
                    all_config_list.append((tar_type, tar_domain, tar_dim, tar_idx, sample_idx))
    return all_config_list


def eval_baseline_init(ins_dir: Union[Path, str] = "../../data/problem_instance",
                       save_init_dir: Union[Path, str] = Path(os.path.dirname(os.path.realpath(__file__)),
                                                              "../../out/svmss_generated_sols"),
                       result_dir: Union[Path, str] = Path(os.path.dirname(os.path.realpath(__file__)),
                                                           "../../out/eval_result"), baseline: str = "SVMSS",
                       tar_type: str = "test", ea_name: str = "elite_ga", sample_batch_size: int = 3,
                       init_batch_size: int = 10, init_eval_times: int = 88, save_result: bool = True, ):
    if not os.path.exists(save_init_dir):
        os.makedirs(save_init_dir)
    if save_result:
        result_name = f"{tar_type}-{ea_name}-{baseline}-{init_eval_times}.pkl"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if os.path.exists(Path(result_dir, result_name)):
            return
    data_worker_num, result_worker_num, clean_worker_num = 10, 10, 1
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
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    all_config_list = generate_eval_data(
        tar_domains=tar_domains, tar_idx_num=tar_idx_num, sample_batch_size=sample_batch_size, tar_type=tar_type
    )
    ea_results = {}
    for task_idx in range(len(all_config_list)):
        configs = all_config_list[task_idx]
        up_queue.put(
            (configs, ins_dir, save_init_dir, init_batch_size, init_eval_times, ea_name, baseline, task_idx)
        )
    while len(ea_results) < len(all_config_list):
        task_idx, metric = score_queue.get()
        ea_results[task_idx] = {
            "config": all_config_list[task_idx],
            "metric": metric
        }
    if save_result:
        result_name = f"{tar_type}-{ea_name}-{baseline}-{init_eval_times}.pkl"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        open(Path(result_dir, result_name), "wb").write(pickle.dumps(ea_results))
    [worker.terminate() for worker in worker_list]

if __name__ == "__main__":
    ins_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    result_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/eval_result")
    ea_names = ["brkga", "elite_ga"]
    baselines = ["KAES", "SVMSS", "OBL", "QI"]
    max_init_eval_times = [132]
    max_parallel_num = 1
    p_list = []
    for ea_name in ea_names:
        for baseline_init in baselines:
            for init_eval_time in max_init_eval_times:
                while len(p_list) >= max_parallel_num:
                    for process in p_list:
                        if not process.is_alive():
                            process.join()
                            process.close()
                            p_list.remove(process)
                print(baseline_init, ea_name, init_eval_time)
                p = Process(target=eval_baseline_init,
                            args=(ins_dir, baseline_init_dirs[baseline_init], result_dir, baseline_init, "test",
                                  ea_name, 3, 10, init_eval_time, True))
                p.start()
                p_list.append(p)
                time.sleep(200)
    [p.join() for p in p_list]
