import logging
import shutil
from itertools import chain

import yaml
from scipy.stats import pearsonr, spearmanr, kendalltau
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.experiments.experiment_problem import load_problem_data, load_sample_indices
from src.network.dataset import SolutionMappingData
from src.network.surrogate_vae import SurrogateVAE
from src.types_ import *


class DecoderMapping:
    def __init__(self, ins_dir: Union[Path, str] = '../../data/problem_instance',
                 surrogate_dir: Union[Path, str] = '../../out/surrogate_models',
                 map_dir: Union[Path, str] = '../../out/map_surrogate_models',
                 src_domain: str = "match_max_problem", tar_domain: str = "match_max_problem",
                 src_type: str = "train", tar_type: str = "gate_train", src_dim: int = 30, tar_dim: int = 30,
                 src_idx: int = 0, tar_idx: int = 0, sample_num: int = 64, src_coefficient: int = 4,
                 sample_idx: int = 0):
        self.ins_dir = Path(ins_dir)
        self.surrogate_dir = Path(surrogate_dir)
        self.map_dir = Path(map_dir)
        self.src_domain = src_domain
        self.tar_domain = tar_domain
        self.src_type = src_type
        self.tar_type = tar_type
        self.src_dim = src_dim
        self.tar_dim = tar_dim
        self.src_idx = src_idx
        self.tar_idx = tar_idx
        self.sample_idx = sample_idx
        self.sample_num = sample_num
        self.src_coefficient = src_coefficient
        # self.src_model = self.load_src_surrogate()
        self.map_model = self.load_src_surrogate()

    def load_src_surrogate(self) -> SurrogateVAE:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src_model_dir = Path(self.surrogate_dir, f"{self.src_domain}_{self.src_dim}_{self.src_idx}")
        with open(Path(src_model_dir, "HyperParam.yaml"), 'r') as file:
            config = yaml.safe_load(file)
        config["model_params"]["out_dim"] = self.tar_dim
        model = SurrogateVAE(**config['model_params']).to(device)
        weight_param = torch.load(str(Path(src_model_dir, "best_model.pt")), map_location=device)
        if self.src_dim == self.tar_dim:
            model.load_state_dict(weight_param)
        else:
            weight_param = {k: v.to(device) for k, v in weight_param.items() if "final_layer" not in k}
            model_dict = model.state_dict()
            model_dict.update(weight_param)
            model.load_state_dict(model_dict)
        return model.to(device)

    def load_map_model(self) -> bool:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(Path(os.path.dirname(os.path.realpath(__file__)), "../../configs/surrogate_mapping.yaml"),
                  'r') as file:
            config = yaml.safe_load(file)
        config["logging_params"]["name"] = "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
            self.src_type, self.src_domain, self.src_dim, self.src_idx, self.tar_type, self.tar_domain, self.tar_dim,
            self.tar_idx, self.sample_num, self.src_coefficient, self.sample_idx
        )
        map_model_dir = Path(self.map_dir, config["logging_params"]["name"])
        if not os.path.exists(Path(map_model_dir, "best_model.pt")):
            return False
        with open(Path(map_model_dir, "HyperParam.yaml"), 'r') as file:
            config = yaml.safe_load(file)
        config["model_params"]["out_dim"] = self.tar_dim
        model = SurrogateVAE(**config['model_params']).to(device)
        model.load_state_dict(torch.load(str(Path(map_model_dir, "best_model.pt")), map_location=device))
        self.map_model = model.to(device)
        return True

    def sample_src_data(self) -> Tuple[NpArray, NpArray]:
        src_ins_dir = Path(self.ins_dir, self.src_type, f"{self.src_domain}_{self.src_dim}_{self.src_idx}")
        src_x, src_y = load_problem_data(problem_dir=src_ins_dir)
        src_indices = load_sample_indices(problem_dir=src_ins_dir, sample_num=self.sample_num * self.src_coefficient,
                                          indices_idx=self.sample_idx)
        return src_x[src_indices], src_y[src_indices]

    def sample_tar_data(self) -> Tuple[NpArray, NpArray]:
        tar_ins_dir = Path(self.ins_dir, self.tar_type, f"{self.tar_domain}_{self.tar_dim}_{self.tar_idx}")
        tar_x, tar_y = load_problem_data(problem_dir=tar_ins_dir)
        tar_indices = load_sample_indices(problem_dir=tar_ins_dir, sample_num=self.sample_num,
                                          indices_idx=self.sample_idx)
        return tar_x[tar_indices], tar_y[tar_indices]

    def generate_cartesian_mapping_data(self) -> Tuple[NpArray, NpArray]:
        src_x, src_y = self.sample_src_data()
        tar_x, tar_y = self.sample_tar_data()
        tar_rank_indices = np.argsort(tar_y)[::-1]
        src_rank_indices = np.argsort(src_y)[::-1]
        src_qual_sols, tar_qual_sols = {}, {}
        for index in range(len(src_y)):
            if src_y[src_rank_indices[index]] not in src_qual_sols:
                src_qual_sols[src_y[src_rank_indices[index]]] = []
            src_qual_sols[src_y[src_rank_indices[index]]].append(src_x[src_rank_indices[index]])
        for index in range(len(tar_y)):
            if tar_y[tar_rank_indices[index]] not in tar_qual_sols:
                tar_qual_sols[tar_y[tar_rank_indices[index]]] = []
            tar_qual_sols[tar_y[tar_rank_indices[index]]].append(tar_x[tar_rank_indices[index]])
        rank_level_num = min(len(src_qual_sols), len(tar_qual_sols))
        src_quals, tar_quals = list(src_qual_sols.keys()), list(tar_qual_sols.keys())
        src_quals.sort(reverse=True)
        tar_quals.sort(reverse=True)

        def fill_level_sols(qual_sols: Dict, level_num: int):
            quals = list(qual_sols.keys())
            quals.sort()
            if len(quals) == level_num:
                return {i: qual_sols[quals[i]] for i in range(level_num)}
            levels = {i: [] for i in range(level_num)}
            data_size = sum([len(qual_sols[qual]) for qual in quals])
            avg_size = data_size / level_num
            current_level_idx = 0
            accumulate_size = 0
            for index, qual in enumerate(quals):
                if len(levels[current_level_idx]) == 0:
                    levels[current_level_idx] += qual_sols[qual]
                    accumulate_size += len(qual_sols[qual])
                    continue
                elif len(quals) - index < level_num - current_level_idx and current_level_idx < level_num - 1:
                    current_level_idx += 1
                if (avg_size * (current_level_idx + 1) - accumulate_size <= 2 * len(qual_sols[qual]) // 3
                        and current_level_idx < level_num - 1):
                    current_level_idx += 1
                levels[current_level_idx] += qual_sols[qual]
                accumulate_size += len(qual_sols[qual])
            return levels

        in_levels = fill_level_sols(src_qual_sols, rank_level_num)
        out_levels = fill_level_sols(tar_qual_sols, rank_level_num)

        src_in, tar_out = [], []
        for level in range(rank_level_num):
            for src_sol in in_levels[level]:
                for out_sol in out_levels[level]:
                    src_in.append(src_sol)
                    tar_out.append(out_sol)
        src_in, tar_out = np.array(src_in, dtype=np.float32), np.array(tar_out, dtype=np.float32)
        # print(len(src_in), len(tar_out))
        indices = np.arange(len(src_in))
        np.random.shuffle(indices)
        return src_in[indices], tar_out[indices]

    def generate_cartesian_mapping_data_top_qual(self) -> Tuple[NpArray, NpArray]:
        src_x, src_y = self.sample_src_data()
        tar_x, tar_y = self.sample_tar_data()
        tar_rank_indices = np.argsort(tar_y)[::-1]
        src_rank_indices = np.argsort(src_y)[::-1]
        src_qual_sols, tar_qual_sols = {}, {}
        for index in range(len(src_y)):
            if src_y[src_rank_indices[index]] not in src_qual_sols:
                src_qual_sols[src_y[src_rank_indices[index]]] = []
            src_qual_sols[src_y[src_rank_indices[index]]].append(src_x[src_rank_indices[index]])
        for index in range(len(tar_y)):
            if tar_y[tar_rank_indices[index]] not in tar_qual_sols:
                tar_qual_sols[tar_y[tar_rank_indices[index]]] = []
            tar_qual_sols[tar_y[tar_rank_indices[index]]].append(tar_x[tar_rank_indices[index]])
        rank_level_num = min(len(src_qual_sols), len(tar_qual_sols))
        src_qual_levels, tar_qual_levels = list(src_qual_sols.keys()), list(tar_qual_sols.keys())
        src_qual_levels.sort(reverse=True)
        tar_qual_levels.sort(reverse=True)
        src_in, tar_out = [], []
        for index in range(rank_level_num):
            for rel_solution in src_qual_sols[src_qual_levels[index]]:
                for eval_solution in tar_qual_sols[tar_qual_levels[index]]:
                    src_in.append(rel_solution)
                    tar_out.append(eval_solution)
        src_in, tar_out = np.array(src_in, dtype=np.float32), np.array(tar_out, dtype=np.float32)
        # print(len(src_in), len(tar_out))
        indices = np.arange(len(src_in))
        np.random.shuffle(indices)
        return src_in[indices], tar_out[indices]

    def fine_tuning_mapping_decoder(self):
        logging.disable(logging.INFO)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src_in, tar_out = self.generate_cartesian_mapping_data()
        with open(Path(os.path.dirname(os.path.realpath(__file__)), "../../configs/surrogate_mapping.yaml"),
                  'r') as file:
            config = yaml.safe_load(file)
        config["model_params"]["in_dim"] = self.src_dim
        config["model_params"]["out_dim"] = self.tar_dim
        config["logging_params"]["name"] = "{}_{}_{}_{}/{}_{}_{}_{}/{}_{}/{}".format(
            self.src_type, self.src_domain, self.src_dim, self.src_idx, self.tar_type, self.tar_domain, self.tar_dim,
            self.tar_idx, self.sample_num, self.src_coefficient, self.sample_idx
        )
        seed_everything(config['exp_params']['manual_seed'], True)
        train_dataloader = DataLoader(SolutionMappingData(src_in, tar_out, "all"),
                                      batch_size=config['data_params']['train_batch_size'], shuffle=True,
                                      num_workers=config['data_params']['num_workers'])
        log_path = Path(config['logging_params']['save_dir'], config["logging_params"]["name"])
        if os.path.exists(log_path):
            shutil.rmtree(log_path)
        os.makedirs(log_path)
        np.save(str(Path(log_path, "source_input.npy")), src_in)
        np.save(str(Path(log_path, "target_output.npy")), tar_out)
        writer = SummaryWriter(str(log_path))
        yaml.dump(config, open(Path(log_path, "HyperParam.yaml"), "w"))
        for (name, param) in self.map_model.named_parameters():
            param.requires_grad = False
        for (name, param) in self.map_model.decoder.named_parameters():
            param.requires_grad = True
        for (name, param) in self.map_model.final_layer.named_parameters():
            param.requires_grad = True
        for (name, param) in self.map_model.decoder_input.named_parameters():
            param.requires_grad = True
        optimizer = optim.Adam(
            chain(self.map_model.decoder.parameters(), self.map_model.final_layer.parameters(),
                  self.map_model.decoder_input.parameters()), lr=config['exp_params']['LR'],
            weight_decay=config['exp_params']['weight_decay'])
        # epoch_bar = tqdm(range(int(config['trainer_params']['max_epochs'])))
        best_val_loss = np.inf
        best_state_dict = self.map_model.state_dict()
        # for epoch in epoch_bar:
        for epoch in range(int(config['trainer_params']['max_epochs'])):
            loss_records = {}
            for ref_solution, eval_solution in train_dataloader:
                if len(ref_solution) == 1:
                    continue
                optimizer.zero_grad()
                train_loss = self.map_model.mapping_loss_function(ref_solution.to(device), eval_solution.to(device))
                train_loss['loss'].backward()
                optimizer.step()
                for key in train_loss.keys():
                    if key not in loss_records:
                        loss_records[key] = []
                    loss_records[key].append(
                        train_loss[key] if key != "loss" else train_loss[key].cpu().detach().numpy())
            if np.mean(loss_records['loss']) < best_val_loss:
                best_val_loss = np.mean(loss_records['loss'])
                best_state_dict = self.map_model.state_dict()

            for key in loss_records.keys():
                writer.add_scalar(key, np.mean(loss_records[key]), epoch)

            # epoch_bar.set_description("Epoch {}".format(epoch))
            # epoch_bar.set_postfix_str("MSE {:.5f}".format(np.mean(loss_records['loss'])))
        torch.save(best_state_dict, Path(log_path, "best_model.pt"))
        # print("Finish the surrogate Task of {}_{}_{} to {}_{}_{}".format(self.src_domain, self.src_dim, self.src_idx,
        #                                                                  self.tar_domain, self.tar_dim, self.tar_idx))
        return (config, open(Path(log_path, "best_model.pt"), "rb").read(), src_in, tar_out)

    def get_correlation(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tar_x, tar_y = self.sample_tar_data()
        if self.tar_dim > self.src_dim:
            tar_x = np.array([x[:self.src_dim] for x in tar_x])
        elif self.tar_dim < self.src_dim:
            tar_x = np.pad(tar_x, pad_width=((0, 0), (0, self.src_dim - self.tar_dim)),
                           mode="constant", constant_values=0)
        pred_y = self.map_model.forward_only_score(torch.from_numpy(tar_x).to(device))
        pred_y = pred_y.cpu().cpu().detach().numpy()
        pearson = pearsonr(tar_y, pred_y)
        rank_y = np.argsort(np.argsort(tar_y)[::-1])
        pred_rank = np.argsort(np.argsort(pred_y)[::-1])
        spearman = spearmanr(np.array(rank_y, dtype=np.int32), np.array(pred_rank, dtype=np.int32))
        kendall = kendalltau(tar_y, pred_y)
        return {
            "pearson": pearson.statistic,
            "spearman": spearman.statistic,
            "kendall": kendall.statistic
        }

    def get_topk_target_solution(self, k=10, source_sample_num=2000000) -> NpArray:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.map_model.eval()
        topk_x, topk_y = [], []
        for _ in range(source_sample_num // 1024):
            solution = torch.randint(0, 2, (1024, self.src_dim), dtype=torch.float32, device=device)
            for _ in range(1):
                target_output, mu, log_var, performance = self.map_model(solution)
                _, indices = torch.topk(performance, k=min(k * 10, 512))
                target_output = target_output[indices].cpu().detach().numpy()
                performance = performance[indices].cpu().detach().numpy()
                target_output = (target_output > 0.5).astype(np.int_)
                target_output, performance = list(target_output) + topk_x, list(performance) + topk_y
                sort_index, solution_str_set = np.argsort(performance)[::-1], set()
                topk_x, topk_y = [], []
                for index in sort_index:
                    solution_str = "".join([str(bit) for bit in target_output[index]])
                    if len(solution_str_set) >= k:
                        break
                    if solution_str not in solution_str_set:
                        solution_str_set.add(solution_str)
                        topk_x.append(target_output[index])
                        topk_y.append(performance[index])
        return np.array(topk_x)
