from multiprocessing import Process

import requests
import yaml

from src.distribution.util import random_str
from src.problem_domain import train_problem_domains, valid_problem_domains, problem_settings
from src.types_ import *
from src.experiments_setting import master_host


def send_surrogate_map_task(ins_dir: Union[Path, str] = '../../data/problem_instance',
                            surrogate_dir: Union[Path, str] = '../../out/surrogate_models',
                            map_dir: Union[Path, str] = '../../out/map_surrogate_models',
                            src_domain: str = "match_max_problem", tar_domain: str = "match_max_problem",
                            src_type: str = "train", tar_type: str = "gate_train", src_dim: int = 30, tar_dim: int = 30,
                            src_idx: int = 0, tar_idx: int = 0, sample_num: int = 64, src_coefficient: int = 4,
                            sample_idx: int = 0):
    task_id = f"{str(int(time.time() * 1000))}_{random_str(32)}"
    task_args = (
        ins_dir, surrogate_dir, map_dir, src_domain, tar_domain, src_type, tar_type, src_dim, tar_dim,
        src_idx, tar_idx, sample_num, src_coefficient, sample_idx
    )
    task = {
        "task_id": task_id,
        "task_func": "eval_surrogate_mapping",
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
    map_model_dir = Path(map_dir, "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
        "train", src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim,
        tar_idx, sample_num, src_coefficient, sample_idx
    ))
    yaml.dump(result[0], open(Path(map_model_dir, "HyperParam.yaml"), "w"))
    open(Path(map_model_dir, "best_model.pt"), "wb").write(result[1])
    np.save(str(Path(map_model_dir, "source_input.npy")), result[2])
    np.save(str(Path(map_model_dir, "target_output.npy")), result[3])
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


def map_all_surrogate(surrogate_dir: Union[Path, str] = "../../out/surrogate_models",
                      ins_dir: Union[Path, str] = "../../data/problem_instance",
                      map_dir: Union[Path, str] = "../../out/map_surrogate_models",
                      tar_type: str = "gate_train", sample_num: int = 64, src_coefficient: int = 4,
                      sample_batch_size: int = 3):
    max_parallel_num = 128
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
    p_list = []
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    for src_domain in train_problem_domains.keys():
        for src_dim in problem_settings["training_dims"]:
            for src_idx in range(problem_settings["training_ins_num"]):
                for tar_domain in tar_domains.keys():
                    for tar_dim in problem_settings["valid_dims"]:
                        for tar_idx in range(tar_idx_num):
                            for sample_idx in range(sample_batch_size):
                                map_model_dir = Path(map_dir, "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
                                    "train", src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim,
                                    tar_idx, sample_num, src_coefficient, sample_idx
                                ))
                                if not os.path.exists(map_model_dir):
                                    os.makedirs(map_model_dir)
                                if os.path.exists(Path(map_model_dir, "best_model.pt")) and os.path.exists(
                                        Path(map_model_dir, "HyperParam.yaml")):
                                    continue
                                while len(p_list) >= max_parallel_num:
                                    for process in p_list:
                                        if not process.is_alive():
                                            process.join()
                                            process.close()
                                            p_list.remove(process)

                                p = Process(
                                    target=send_surrogate_map_task,
                                    args=(
                                        ins_dir, surrogate_dir, map_dir, src_domain, tar_domain, "train", tar_type,
                                        src_dim, tar_dim, src_idx, tar_idx, sample_num, src_coefficient, sample_idx
                                    ))
                                p_list.append(p)
                                p.start()
                                time.sleep(0.1)
    [p.join() for p in p_list]


if __name__ == '__main__':
    work_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    surrogate_model_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/surrogate_models")
    map_model_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/map_surrogate_models")
    map_all_surrogate(map_dir=map_model_dir, ins_dir=work_dir, surrogate_dir=surrogate_model_dir,
                      tar_type="gate_train", sample_num=64, src_coefficient=4, sample_batch_size=3)
    map_all_surrogate(map_dir=map_model_dir, ins_dir=work_dir, surrogate_dir=surrogate_model_dir,
                      tar_type="test", sample_num=64, src_coefficient=4, sample_batch_size=3)