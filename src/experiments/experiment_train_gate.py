import json
import logging
import shutil
import zipfile
from io import BytesIO
from multiprocessing import Process, Queue

import logzero
import requests

from src.distribution.util import random_str
from src.ea import ea_algs
from src.es.pgpe import PGPE
from src.experiments.experiment_problem import load_problem_data, load_sample_indices
from src.experiments_setting import master_host
from src.network.gate_net import GateNet
from src.problem_domain import train_problem_domains, valid_problem_domains, problem_cost, problem_settings
from src.types_ import *


def average_hamming_distance(sols):
    sols = np.array(sols)
    distances = []
    for array in sols:
        distances.append(np.mean(np.abs([sol - array for sol in sols])))
    avg_distance = np.mean(np.array(distances))
    return avg_distance



def generate_train_data(corr_dict, tar_domains, tar_idx_num, sample_batch_size, src_type, tar_type, sample_num,
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
            "task_func": "eval_problem_EA",
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
            qual_dict = {}
            with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.is_dir(): continue
                    with zip_ref.open(file_info) as file:
                        file_data = file.read()
                        result = pickle.loads(file_data)
                        qual_dict[file_info.filename] = result[1]
            if len(qual_dict) != len(all_task_id):
                print("GET RESULT ERROR!!!")
                result_queue.put(data)
            else:
                all_quals = [qual_dict[task_id] for task_id in all_task_id]
                metric = (np.mean(all_quals) - min_y) / (max_y - min_y)
                score_queue.put((task_idx, metric))
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


def eval_config_on_ins_worker(up_queue: Queue, result_queue: Queue, score_queue: Queue):
    while True:
        data = up_queue.get()
        configs, ins_dir, map_dir, init_batch_size, k, ea_name, task_idx = data
        init_sol_dir = Path(map_dir, "../map_generated_sols")
        all_task_args, all_generate_qual_records, all_generate_sols_diversity = [], [], []
        (src_type, src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim, tar_idx,
         sample_num, src_coefficient, sample_idx) = configs[0]
        tar_ins_dir = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
        tar_x, tar_y = load_problem_data(problem_dir=tar_ins_dir)
        tar_indices = load_sample_indices(problem_dir=tar_ins_dir, sample_num=sample_num, indices_idx=sample_idx)
        min_y, max_y = np.min(tar_y), np.max(tar_y)
        for idx in range(init_batch_size):
            generate_sols = []
            generate_quals = []
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
                generate_sols += sols
                generate_quals += quals
            init_sols = generate_sols + list(tar_x[tar_indices])
            init_quals = generate_quals + list(tar_y[tar_indices])
            if ea_name not in ["no_ea", "no_ea_diversity", "no_ea_max", "no_ea_max_diversity"]:
                init_idx = np.argsort(np.array(init_quals))[::-1]
                init_sols = np.array(init_sols, dtype=np.int64)[init_idx]
                ea = ea_algs[ea_name]
                ea_args = ea.recommend_kwargs.copy()
                ea_args["initial_solutions"] = init_sols
                task_args = (ins_dir, tar_domain, tar_type, tar_dim, tar_idx, ea_name, ea_args)
                all_task_args.append(task_args)
            else:
                qual_rank = np.argsort(np.array(generate_quals))[::-1]
                if ea_name =="no_ea":
                    all_generate_qual_records.append(np.mean(np.array(generate_quals)[qual_rank]))
                elif ea_name == "no_ea_max":
                    all_generate_qual_records.append(np.max(np.array(generate_quals)[qual_rank]))
                elif ea_name == "no_ea_diversity":
                    all_generate_qual_records.append(np.mean(np.array(generate_quals)[qual_rank]))
                    hamming_dis = average_hamming_distance(np.array(generate_sols)[qual_rank])
                    all_generate_sols_diversity.append(np.mean(hamming_dis))
                elif ea_name == "no_ea_max_diversity":
                    all_generate_qual_records.append(np.max(np.array(generate_quals)[qual_rank]))
                    hamming_dis = average_hamming_distance(np.array(generate_sols)[qual_rank])
                    all_generate_sols_diversity.append(np.mean(hamming_dis))
                else:
                    print("EA NAME ERROR")
        if ea_name == "no_ea":
            avg_mean = (np.mean(all_generate_qual_records) - min_y) / (max_y - min_y)
            metric = avg_mean
            score_queue.put((task_idx, metric))
        elif ea_name == "no_ea_max":
            avg_max = (np.mean(all_generate_qual_records) - min_y) / (max_y - min_y)
            metric = avg_max
            score_queue.put((task_idx, metric))
        elif ea_name == "no_ea_diversity":
            avg_mean = (np.mean(all_generate_qual_records) - min_y) / (max_y - min_y)
            avg_diversity = np.mean(all_generate_sols_diversity)
            metric = (avg_mean + avg_diversity) / 2
            score_queue.put((task_idx, metric))
        elif ea_name == "no_ea_max_diversity":
            avg_max = (np.mean(all_generate_qual_records) - min_y) / (max_y - min_y)
            avg_diversity = np.mean(all_generate_sols_diversity)
            metric = (avg_max + avg_diversity) / 2
            score_queue.put((task_idx, metric))
        else:
            send_ea_eval_tasks(all_task_args, min_y, max_y, task_idx, result_queue)


def train_gate_net(corr_dir: Union[Path, str] = "../../out/correlation_results",
                   ins_dir: Union[Path, str] = "../../data/problem_instance",
                   map_dir: Union[Path, str] = "../../out/map_surrogate_models",
                   weight_dir: Union[Path, str] = "../../out/gate_weights",
                   src_type: str = "train", tar_type: str = "gate_train", ea_name: str = "elite_ga",
                   sample_num: int = 64, src_coefficient: int = 4, sample_batch_size: int = 3,
                   init_batch_size: int = 10, epoch_num: int = 200, pop_size: int = 101, b: int = 12, k: int = 2,
                   load_from_log: str = None):
    save_path = Path(weight_dir, f"{ea_name}-{b}-{k}")
    if Path(save_path, "es.pkl").exists() and Path(save_path, f"train_{ea_name}.log").exists() and \
            Path(save_path, f"train_{ea_name}.npy"):
        print("There is a finished gate weights.", save_path)
        return
    elif save_path.exists():
        shutil.rmtree(save_path)
    data_worker_num, result_worker_num, clean_worker_num = 40, 10, 1
    up_queue, score_queue, result_queue, clean_queue = Queue(), Queue(), Queue(), Queue()
    worker_list: List[Process] = []
    for _ in range(data_worker_num):
        worker = Process(target=eval_config_on_ins_worker, args=(up_queue, result_queue, score_queue))
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
    date_style = '%Y-%m-%d %H:%M:%S'
    file_format = '[%(asctime)s| %(levelname)s |%(filename)s:%(lineno)d] %(message)s'
    formatter = logging.Formatter(file_format, date_style)
    if load_from_log:
        log_path = Path(os.path.dirname(os.path.abspath(__file__)),
                        f"../../logs/train_gate/{load_from_log}")
    else:
        log_path = Path(os.path.dirname(os.path.abspath(__file__)),
                        f"../../logs/train_gate/train_{ea_name}_{int(time.time())}")
        if not log_path.exists():
            os.makedirs(log_path)
    logger = logzero.setup_logger(logfile=str(Path(log_path, f"train_{ea_name}.log")),
                                  formatter=formatter,
                                  name=f"Train Log for {ea_name}",
                                  level=logzero.ERROR, fileLoglevel=logzero.INFO, backupCount=100,
                                  maxBytes=int(1e7))
    if not load_from_log:
        logger.info(f"Start with {data_worker_num} Data Worker, {result_worker_num} Result Worker, "
                    f"{clean_worker_num} Clean Worker")
        logger.info("The Experiment Setting is:")
        logger.info(json.dumps({
            "src_type": src_type,
            "tar_type": tar_type,
            "ea_name": ea_name,
            "sample_num": sample_num,
            "src_coefficient": src_coefficient,
            "sample_batch_size": sample_batch_size,
            "init_batch_size": init_batch_size,
            "epoch_num": epoch_num,
            "pop_size": pop_size,
            "b": b,
            "k": k
        }, ensure_ascii=False, indent=4))

    corr_dict: Dict = pickle.loads(open(Path(corr_dir, f"{src_type}-CORR-{tar_type}-64_4.pkl"), "rb").read())
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    all_input, all_config_list = generate_train_data(
        corr_dict=corr_dict, tar_domains=tar_domains, tar_idx_num=tar_idx_num, sample_batch_size=sample_batch_size,
        src_type=src_type, tar_type=tar_type, sample_num=sample_num, src_coefficient=src_coefficient
    )
    train_ins_num = len(tar_domains) * len(problem_settings["training_dims"]) * tar_idx_num
    gate_net = GateNet(feature_num=train_ins_num * 3, candidate_num=train_ins_num, hidden_dims=[128, 128])
    if load_from_log is None:
        es = PGPE(num_params=gate_net.weights_num, popsize=pop_size, mu=None)
        start_epoch_idx = 0
    else:
        es = pickle.loads(open(Path(log_path, f"es.pkl"), "rb").read())
        start_epoch_idx = es.epoch_idx
    best_weight, best_score = None, -1e15
    for epoch in tqdm(range(start_epoch_idx, epoch_num)):
        weights = es.ask()
        ea_results = {}
        for weight_idx, weight in enumerate(weights):
            results = gate_net(torch.DoubleTensor(all_input), torch.DoubleTensor(weight))
            for ins_idx in range(len(results)):
                top_k_src = np.argsort(results[ins_idx])[::-1][:b]
                configs = [all_config_list[ins_idx][idx] for idx in top_k_src]
                up_queue.put((configs, ins_dir, map_dir, init_batch_size, k, ea_name, f"{weight_idx}_{ins_idx}"))
        while len(ea_results) < len(weights) * len(all_input):
            task_idx, metric = score_queue.get()
            ea_results[task_idx] = metric
        weight_score = [
            np.mean([ea_results[f"{weight_idx}_{ins_idx}"] for ins_idx in range(len(all_input))]) for weight_idx in
            range(len(weights))
        ]
        es.tell(weight_score)
        best_idx = np.argmax(weight_score)
        if weight_score[best_idx] > best_score:
            best_weight, best_score = weights[best_idx], weight_score[best_idx]
            np.save(Path(log_path, f"train_{ea_name}.npy"), best_weight)
        logger.info(f"Epoch {epoch}: max {np.max(weight_score):.4f}\tmin {np.min(weight_score):.4f}\t"
                    f"mean {np.mean(weight_score):.4f}. Current Best: {best_score:.4f}")
        es.epoch_idx = epoch
        open(Path(log_path, f"es.pkl"), "wb").write(pickle.dumps(es))
    shutil.copytree(log_path, save_path)
    [worker.terminate() for worker in worker_list]


if __name__ == '__main__':
    corr_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/correlation_results")
    ins_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    map_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/map_surrogate_models")
    weight_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/gate_weights")
    max_parallel_num = 1
    p_list = []
    ks = [4]
    eas = ["no_ea", "no_ea_diversity", "no_ea_max", "no_ea_max_diversity"]
    for ea_name in eas:
        for k in ks:
            while len(p_list) >= max_parallel_num:
                for process in p_list:
                    if not process.is_alive():
                        process.join()
                        process.close()
                        p_list.remove(process)
            p = Process(target=train_gate_net,
                        args=(corr_dir, ins_dir, map_dir, weight_dir, "train", "gate_train", ea_name, 64, 4, 3, 10, 500,
                              101, 12, k, None))
            p.start()
            p_list.append(p)
            time.sleep(2)
    [p.join() for p in p_list]
