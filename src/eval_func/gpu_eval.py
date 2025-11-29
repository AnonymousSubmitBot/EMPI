import logging
import shutil

import torch.cuda
import yaml
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.experiments.experiment_problem import load_problem_data
from src.network.dataset import ZeroOneProblemData
from src.network.decoder_mapping import DecoderMapping
from src.network.surrogate_vae import SurrogateVAE
from src.problem_domain import BaseProblem
from src.types_ import *


def fit_surrogate(problem_domain: str, ins_dir: Union[Path, str] = '../../data/problem_instance',
                  ins_type: str = "train", dim: int = 30, index: int = 0):
    logging.disable(logging.INFO)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open(Path(os.path.dirname(os.path.realpath(__file__)), "../../configs/surrogate.yaml"), 'r') as file:
        config = yaml.safe_load(file)
    problem_path = Path(ins_dir, ins_type, f"{problem_domain}_{dim}_{index}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))

    config["model_params"]["in_dim"] = problem_instance.dimension
    config["logging_params"]["name"] = f"{problem_domain}_{dim}_{index}"
    seed_everything(config['exp_params']['manual_seed'], True, verbose=False)
    model = SurrogateVAE(**config["model_params"]).to(device)

    x, y = load_problem_data(problem_path)
    train_data = ZeroOneProblemData(x, y, 'train')
    valid_data = ZeroOneProblemData(x, y, 'valid')
    train_dataloader = DataLoader(train_data, batch_size=config['data_params']['train_batch_size'], shuffle=True,
                                  num_workers=config['data_params']['num_workers'])
    valid_dataloader = DataLoader(valid_data, batch_size=config['data_params']['val_batch_size'], shuffle=True,
                                  num_workers=config['data_params']['num_workers'])
    log_path = Path(config['logging_params']['save_dir'], config["logging_params"]["name"])
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    writer = SummaryWriter(str(log_path))
    yaml.dump(config, open(Path(log_path, "HyperParam.yaml"), "w"))
    optimizer = optim.Adam(model.parameters(),
                           lr=config['exp_params']['LR'],
                           weight_decay=config['exp_params']['weight_decay'])
    best_val_loss = np.inf
    for epoch in range(int(config['trainer_params']['max_epochs'])):
        loss_records = {}
        for solution, quality in train_dataloader:
            optimizer.zero_grad()
            train_loss = model.loss_function(solution.to(device), quality.to(device))
            train_loss['loss'].backward()
            optimizer.step()
            for key in train_loss.keys():
                if key not in loss_records:
                    loss_records[key] = []
                loss_records[key].append(train_loss[key] if key != "loss" else train_loss[key].cpu().detach().numpy())
        for solution, quality in valid_dataloader:
            valid_loss = model.loss_function(solution.to(device), quality.to(device))
            for key in valid_loss.keys():
                if "val_{}".format(key) not in loss_records:
                    loss_records["val_{}".format(key)] = []
                loss_records["val_{}".format(key)].append(
                    valid_loss[key] if key != "loss" else valid_loss[key].cpu().detach().numpy())
        if np.mean(loss_records['val_loss']) < best_val_loss:
            best_val_loss = np.mean(loss_records['val_loss'])
            torch.save(model.state_dict(), Path(log_path, "best_model.pt"))

        for key in loss_records.keys():
            writer.add_scalar(key, np.mean(loss_records[key]), epoch)
    return (config, open(Path(log_path, "best_model.pt"), "rb").read())


def map_decoder(ins_dir: Union[Path, str] = '../../data/problem_instance',
                surrogate_dir: Union[Path, str] = '../../out/surrogate_models',
                map_dir: Union[Path, str] = '../../out/map_surrogate_models',
                src_domain: str = "match_max_problem", tar_domain: str = "match_max_problem",
                src_type: str = "train", tar_type: str = "gate_train", src_dim: int = 30, tar_dim: int = 30,
                src_idx: int = 0, tar_idx: int = 0, sample_num: int = 64, src_coefficient: int = 4,
                sample_idx: int = 0):
    dp = DecoderMapping(
        ins_dir=ins_dir, surrogate_dir=surrogate_dir, map_dir=map_dir,
        src_domain=src_domain, tar_domain=tar_domain, src_type=src_type, tar_type=tar_type,
        src_dim=src_dim, tar_dim=tar_dim, src_idx=src_idx, tar_idx=tar_idx,
        sample_num=sample_num, src_coefficient=src_coefficient, sample_idx=sample_idx
    )
    return dp.fine_tuning_mapping_decoder()

def calculate_correlation(ins_dir: Union[Path, str] = '../../data/problem_instance',
                          surrogate_dir: Union[Path, str] = '../../out/surrogate_models',
                          map_dir: Union[Path, str] = '../../out/map_surrogate_models',
                          src_domain: str = "match_max_problem", tar_domain: str = "match_max_problem",
                          src_type: str = "train", tar_type: str = "gate_train", src_dim: int = 30, tar_dim: int = 30,
                          src_idx: int = 0, tar_idx: int = 0, sample_num: int = 64, src_coefficient: int = 4,
                          sample_idx: int = 0):
    dp = DecoderMapping(
        ins_dir=ins_dir, surrogate_dir=surrogate_dir, map_dir=map_dir,
        src_domain=src_domain, tar_domain=tar_domain, src_type=src_type, tar_type=tar_type,
        src_dim=src_dim, tar_dim=tar_dim, src_idx=src_idx, tar_idx=tar_idx,
        sample_num=sample_num, src_coefficient=src_coefficient, sample_idx=sample_idx
    )
    return dp.get_correlation()


def generate_init_solution(ins_dir: Union[Path, str] = '../../data/problem_instance',
                           surrogate_dir: Union[Path, str] = '../../out/surrogate_models',
                           map_dir: Union[Path, str] = '../../out/map_surrogate_models',
                           src_domain: str = "match_max_problem", tar_domain: str = "match_max_problem",
                           src_type: str = "train", tar_type: str = "gate_train", src_dim: int = 30, tar_dim: int = 30,
                           src_idx: int = 0, tar_idx: int = 0, sample_num: int = 64, src_coefficient: int = 4,
                           sample_idx: int = 0, k: int = 8):
    dp = DecoderMapping(
        ins_dir=ins_dir, surrogate_dir=surrogate_dir, map_dir=map_dir,
        src_domain=src_domain, tar_domain=tar_domain, src_type=src_type, tar_type=tar_type,
        src_dim=src_dim, tar_dim=tar_dim, src_idx=src_idx, tar_idx=tar_idx,
        sample_num=sample_num, src_coefficient=src_coefficient, sample_idx=sample_idx
    )
    dp.load_map_model()
    solutions = dp.get_topk_target_solution(k=k)
    problem_path = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    qualities = [problem_instance.evaluate(solution) for solution in solutions]
    return solutions, np.array(qualities)

