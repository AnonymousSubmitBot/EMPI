from multiprocessing import Process, Manager

import requests

from src.distribution.util import random_str
from src.experiments_setting import master_host
from src.problem_domain import train_problem_domains, valid_problem_domains, problem_settings
from src.types_ import *


def send_correlation_cal_task(ins_dir: Union[Path, str] = '../../data/problem_instance',
                              surrogate_dir: Union[Path, str] = '../../out/surrogate_models',
                              map_dir: Union[Path, str] = '../../out/map_surrogate_models',
                              src_domain: str = "match_max_problem", tar_domain: str = "match_max_problem",
                              src_type: str = "train", tar_type: str = "gate_train", src_dim: int = 30,
                              tar_dim: int = 30, src_idx: int = 0, tar_idx: int = 0, sample_num: int = 64,
                              src_coefficient: int = 4, sample_idx: int = 0):
    task_id = f"{str(int(time.time() * 1000))}_{random_str(32)}"
    task_args = (
        ins_dir, surrogate_dir, map_dir, src_domain, tar_domain, src_type, tar_type, src_dim, tar_dim,
        src_idx, tar_idx, sample_num, src_coefficient, sample_idx
    )
    task = {
        "task_id": task_id,
        "task_func": "calculate_correlation",
        "task_args": task_args,
        "task_cost": 1,
        "task_type": "gpu"
    }
    task_data = pickle.dumps(task)
    response_code = 500
    while response_code > 299 or response_code < 200:
        try:
            response = requests.post(url="http://{}/create_task".format(master_host),
                                     data={"check_val": "81600a92e8416bba7d9fada48e9402a4",
                                           "task_type": "gpu", "task_cost": 1, "task_id": task_id},
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
    return result


def get_correlation(ins_dir: Union[Path, str] = '../../data/problem_instance',
                    surrogate_dir: Union[Path, str] = '../../out/surrogate_models',
                    map_dir: Union[Path, str] = '../../out/map_surrogate_models',
                    src_domain: str = "match_max_problem", tar_domain: str = "match_max_problem",
                    src_type: str = "train", tar_type: str = "gate_train", src_dim: int = 30, tar_dim: int = 30,
                    src_idx: int = 0, tar_idx: int = 0, sample_num: int = 64, src_coefficient: int = 4,
                    sample_idx: int = 0, result_dict: Dict[str, Any] = None):
    result = send_correlation_cal_task(ins_dir=ins_dir, surrogate_dir=surrogate_dir, map_dir=map_dir,
                                       src_domain=src_domain, tar_domain=tar_domain, src_type=src_type,
                                       tar_type=tar_type, src_dim=src_dim, tar_dim=tar_dim, src_idx=src_idx,
                                       tar_idx=tar_idx, sample_num=sample_num, src_coefficient=src_coefficient,
                                       sample_idx=sample_idx)
    map_name = "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
        "train", src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim,
        tar_idx, sample_num, src_coefficient, sample_idx
    )
    result_dict[map_name] = result


def check_corr_map_valid(corr_path: Union[Path, str], src_type: str = "train", tar_type: str = "gate_train",
                         sample_num: int = 64, src_coefficient: int = 4, sample_batch_size: int = 3):
    if not os.path.exists(corr_path):
        return False
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    corr_map = pickle.loads(open(corr_path, "rb").read())
    for src_domain in train_problem_domains.keys():
        for src_dim in problem_settings["training_dims"]:
            for src_idx in range(problem_settings["training_ins_num"]):
                for tar_domain in tar_domains.keys():
                    for tar_dim in problem_settings["valid_dims"]:
                        for tar_idx in range(tar_idx_num):
                            for sample_idx in range(sample_batch_size):
                                map_name = "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
                                    src_type, src_domain, src_dim, src_idx,
                                    tar_type, tar_domain, tar_dim, tar_idx,
                                    sample_num, src_coefficient, sample_idx
                                )
                                if map_name not in corr_map.keys():
                                    return False
    return True


def calculate_all_correlation(surrogate_dir: Union[Path, str] = "../../out/surrogate_models",
                              ins_dir: Union[Path, str] = "../../data/problem_instance",
                              map_dir: Union[Path, str] = "../../out/map_surrogate_models",
                              corr_dir: Union[Path, str] = '../../out/correlation_results',
                              src_type: str = "train", tar_type: str = "gate_train", sample_num: int = 64,
                              src_coefficient: int = 4, sample_batch_size: int = 3):
    max_parallel_num = 128
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    if not os.path.exists(corr_dir):
        os.makedirs(corr_dir)
    corr_path = Path(corr_dir, f"{src_type}-CORR-{tar_type}-{sample_num}_{src_coefficient}.pkl")
    if os.path.exists(corr_path) and check_corr_map_valid(corr_path=corr_path, src_type=src_type, tar_type=tar_type,
                                                          sample_num=sample_num, src_coefficient=src_coefficient,
                                                          sample_batch_size=sample_batch_size):
        return
    manager = Manager()
    result_dict = manager.dict()
    p_list = []
    for src_domain in train_problem_domains.keys():
        for src_dim in problem_settings["training_dims"]:
            for src_idx in range(problem_settings["training_ins_num"]):
                for tar_domain in tar_domains.keys():
                    for tar_dim in problem_settings["valid_dims"]:
                        for tar_idx in range(tar_idx_num):
                            for sample_idx in range(sample_batch_size):
                                while len(p_list) >= max_parallel_num:
                                    for process in p_list:
                                        if not process.is_alive():
                                            process.join()
                                            process.close()
                                            p_list.remove(process)
                                p = Process(
                                    target=get_correlation,
                                    args=(
                                        ins_dir, surrogate_dir, map_dir, src_domain, tar_domain, src_type, tar_type,
                                        src_dim, tar_dim, src_idx, tar_idx, sample_num, src_coefficient, sample_idx,
                                        result_dict
                                    ))
                                p_list.append(p)
                                p.start()
                                time.sleep(0.1)
    [p.join() for p in p_list]
    corr_dict: Dict[str, Dict[str, float]] = {}
    for src_domain in train_problem_domains.keys():
        for src_dim in problem_settings["training_dims"]:
            for src_idx in range(problem_settings["training_ins_num"]):
                for tar_domain in tar_domains.keys():
                    for tar_dim in problem_settings["valid_dims"]:
                        for tar_idx in range(tar_idx_num):
                            for sample_idx in range(sample_batch_size):
                                map_name = "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
                                    src_type, src_domain, src_dim, src_idx,
                                    tar_type, tar_domain, tar_dim, tar_idx,
                                    sample_num, src_coefficient, sample_idx
                                )
                                reverse_name = "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
                                    tar_type, tar_domain, tar_dim, tar_idx,
                                    src_type, src_domain, src_dim, src_idx,
                                    sample_num, src_coefficient, sample_idx
                                )
                                corr_dict[map_name] = result_dict[map_name]
                                corr_dict[reverse_name] = result_dict[map_name]
    open(corr_path, "wb").write(pickle.dumps(corr_dict))


if __name__ == '__main__':
    work_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    surrogate_model_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/surrogate_models")
    map_model_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/map_surrogate_models")
    corr_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/correlation_results")
    calculate_all_correlation(map_dir=map_model_dir, ins_dir=work_dir, surrogate_dir=surrogate_model_dir,
                              corr_dir=corr_dir, tar_type="gate_train", sample_num=64, src_coefficient=4,
                              sample_batch_size=3)
    calculate_all_correlation(map_dir=map_model_dir, ins_dir=work_dir, surrogate_dir=surrogate_model_dir,
                              corr_dir=corr_dir, tar_type="test", sample_num=64, src_coefficient=4, sample_batch_size=3)
