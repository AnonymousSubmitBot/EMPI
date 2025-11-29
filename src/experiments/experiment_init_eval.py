from multiprocessing import Process, Manager

import requests

from src.distribution.util import random_str
from src.experiments_setting import master_host
from src.problem_domain import train_problem_domains, valid_problem_domains, problem_settings
from src.types_ import *


def send_generate_init_task(ins_dir: Union[Path, str] = '../../data/problem_instance',
                            surrogate_dir: Union[Path, str] = '../../out/surrogate_models',
                            map_dir: Union[Path, str] = '../../out/map_surrogate_models',
                            src_domain: str = "match_max_problem", tar_domain: str = "match_max_problem",
                            src_type: str = "train", tar_type: str = "gate_train", src_dim: int = 30, tar_dim: int = 30,
                            src_idx: int = 0, tar_idx: int = 0, sample_num: int = 64, src_coefficient: int = 4,
                            sample_idx: int = 0, k: int = 8):
    task_id = f"{str(int(time.time() * 1000))}_{random_str(32)}"
    task_args = (
        ins_dir, surrogate_dir, map_dir, src_domain, tar_domain, src_type, tar_type, src_dim, tar_dim,
        src_idx, tar_idx, sample_num, src_coefficient, sample_idx, k
    )
    task = {
        "task_id": task_id,
        "task_func": "generate_init_solution",
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


def get_init_solution(ins_dir: Union[Path, str] = '../../data/problem_instance',
                      surrogate_dir: Union[Path, str] = '../../out/surrogate_models',
                      map_dir: Union[Path, str] = '../../out/map_surrogate_models',
                      src_domain: str = "match_max_problem", tar_domain: str = "match_max_problem",
                      src_type: str = "train", tar_type: str = "gate_train", src_dim: int = 30, tar_dim: int = 30,
                      src_idx: int = 0, tar_idx: int = 0, sample_num: int = 64, src_coefficient: int = 4,
                      sample_idx: int = 0, ks=None, result_dict: Dict = None, save_init_sol: bool = False,
                      init_batch_size: int = 10):
    if ks is None:
        ks = [2, 4]
    max_k = max(ks)
    map_name = "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
        "train", src_domain, src_dim, src_idx, tar_type, tar_domain, tar_dim,
        tar_idx, sample_num, src_coefficient, sample_idx
    )
    if save_init_sol:
        for init_idx in range(init_batch_size):
            init_sol_dir = Path(map_dir, "../map_generated_sols")
            save_dir = Path(init_sol_dir, map_name)
            all_exist = True
            for k in ks:
                if os.path.exists(Path(save_dir, f"init_{k}_solutions_{init_idx}.npy")) and os.path.exists(
                        Path(save_dir, f"init_{k}_qualities_{init_idx}.npy")):
                    try:
                        sols = np.load(Path(save_dir, f"init_{k}_solutions_{init_idx}.npy"))
                        quals = np.load(Path(save_dir, f"init_{k}_qualities_{init_idx}.npy"))
                        if sols.shape != (k, tar_dim) or quals.shape != (k,):
                            all_exist = False
                            print(f"Re-get {map_name} {k} {init_idx}")
                            break
                    except Exception:
                        all_exist = False
                        pass
                else:
                    all_exist = False
                    break
            if all_exist:
                continue

            result = send_generate_init_task(ins_dir=ins_dir, surrogate_dir=surrogate_dir, map_dir=map_dir,
                                             src_domain=src_domain, tar_domain=tar_domain, src_type=src_type,
                                             tar_type=tar_type, src_dim=src_dim, tar_dim=tar_dim, src_idx=src_idx,
                                             tar_idx=tar_idx, sample_num=sample_num, src_coefficient=src_coefficient,
                                             sample_idx=sample_idx, k=max_k * 10)
            if not save_dir.exists():
                os.makedirs(save_dir)
            for k in ks:
                np.save(Path(save_dir, f"init_{k}_solutions_{init_idx}.npy"), result[0][:k])
                np.save(Path(save_dir, f"init_{k}_qualities_{init_idx}.npy"), result[1][:k])
    else:
        result = send_generate_init_task(ins_dir=ins_dir, surrogate_dir=surrogate_dir, map_dir=map_dir,
                                         src_domain=src_domain, tar_domain=tar_domain, src_type=src_type,
                                         tar_type=tar_type, src_dim=src_dim, tar_dim=tar_dim, src_idx=src_idx,
                                         tar_idx=tar_idx, sample_num=sample_num, src_coefficient=src_coefficient,
                                         sample_idx=sample_idx, k=max_k * 10)
        result_dict[map_name] = result


def generate_all_init(surrogate_dir: Union[Path, str] = "../../out/surrogate_models",
                      ins_dir: Union[Path, str] = "../../data/problem_instance",
                      map_dir: Union[Path, str] = "../../out/map_surrogate_models",
                      tar_type: str = "gate_train", sample_num: int = 64, src_coefficient: int = 4,
                      sample_batch_size: int = 3, ks=None, save_init_sol: bool = False,
                      init_batch_size: int = 10):
    if ks is None:
        ks = [2, 4]
    max_parallel_num = 128
    p_list = []
    manager = Manager()
    result_dict = manager.dict()
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
                                if not os.path.exists(Path(map_model_dir, "best_model.pt")) and os.path.exists(
                                        Path(map_model_dir, "HyperParam.yaml")):
                                    print("NO MODEL!")
                                    exit()
                                while len(p_list) >= max_parallel_num:
                                    for process in p_list:
                                        if not process.is_alive():
                                            process.join()
                                            process.close()
                                            p_list.remove(process)
                                p = Process(target=get_init_solution,
                                            args=(
                                                ins_dir, surrogate_dir, map_dir, src_domain, tar_domain, "train",
                                                tar_type, src_dim, tar_dim, src_idx, tar_idx, sample_num,
                                                src_coefficient, sample_idx, ks, result_dict, save_init_sol,
                                                init_batch_size
                                            ))
                                p_list.append(p)
                                p.start()
    [p.join() for p in p_list]


if __name__ == '__main__':
    work_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    surrogate_model_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/surrogate_models")
    map_model_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../out/map_surrogate_models")
    generate_all_init(map_dir=map_model_dir, ins_dir=work_dir, surrogate_dir=surrogate_model_dir,
                      tar_type="gate_train", sample_num=64, src_coefficient=4, ks=[4], sample_batch_size=3,
                      save_init_sol=True, init_batch_size=10)
    generate_all_init(map_dir=map_model_dir, ins_dir=work_dir, surrogate_dir=surrogate_model_dir,
                      tar_type="test", sample_num=64, src_coefficient=4, ks=[4], sample_batch_size=3,
                      save_init_sol=True, init_batch_size=10)

