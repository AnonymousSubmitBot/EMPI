from scipy.stats import ranksums

from src.experiments.experiment_problem import load_problem_data
from src.problem_domain import problem_settings, train_problem_domains, valid_problem_domains
from src.types_ import *

ins_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")


def normalize_results(results: Dict) -> Dict:
    for key in results.keys():
        config = results[key]["config"]
        if len(config) == 11:
            (_, _, _, _, tar_type, tar_domain, tar_dim, tar_idx,
             sample_num, _, sample_idx) = config
        else:
            (tar_type, tar_domain, tar_dim, tar_idx, sample_idx) = config
        tar_ins_dir = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
        _, tar_y = load_problem_data(problem_dir=tar_ins_dir)
        min_y, max_y = np.min(tar_y), np.max(tar_y)
        for idx in range(len(results[key]["metric"])):
            results[key]["metric"][idx]["best_y"] = (results[key]["metric"][idx]["best_y"] - min_y) / (max_y - min_y)
            norm_steps = (np.array(results[key]["metric"][idx]["step_history"]) - min_y) / (max_y - min_y)
            results[key]["metric"][idx]["step_history"] = norm_steps
            norm_pops = [(np.array(pop) - min_y) / (max_y - min_y) for pop in
                         results[key]["metric"][idx]["pop_history"]]
            results[key]["metric"][idx]["pop_history"] = norm_pops
    return results


def analysis_step_num(results_1, results_2, step_num: int = 800):
    wdl = [0, 0, 0]
    source_1, source_2 = {}, {}
    tar_type = ""
    for key in results_1.keys():
        config = results_1[key]["config"]
        if len(config) == 11:
            (_, _, _, _, tar_type, tar_domain, tar_dim, tar_idx,
             sample_num, _, sample_idx) = config
        else:
            (tar_type, tar_domain, tar_dim, tar_idx, sample_idx) = config
        top_best_y_1 = [np.max(item["step_history"][:step_num]) for item in results_1[key]["metric"]]
        top_best_y_2 = [np.max(item["step_history"][:step_num]) for item in results_2[key]["metric"]]
        if tar_domain not in source_1.keys():
            source_1[tar_domain] = {}
        if tar_dim not in source_1[tar_domain].keys():
            source_1[tar_domain][tar_dim] = {}
        if tar_idx not in source_1[tar_domain][tar_dim].keys():
            source_1[tar_domain][tar_dim][tar_idx] = []
        if tar_domain not in source_2.keys():
            source_2[tar_domain] = {}
        if tar_dim not in source_2[tar_domain].keys():
            source_2[tar_domain][tar_dim] = {}
        if tar_idx not in source_2[tar_domain][tar_dim].keys():
            source_2[tar_domain][tar_dim][tar_idx] = []
        source_1[tar_domain][tar_dim][tar_idx] += top_best_y_1
        source_2[tar_domain][tar_dim][tar_idx] += top_best_y_2
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    large_mean = 0
    for tar_domain in tar_domains.keys():
        for tar_dim in problem_settings["valid_dims"]:
            for tar_idx in range(tar_idx_num):
                if ranksums(source_1[tar_domain][tar_dim][tar_idx], source_2[tar_domain][tar_dim][tar_idx],
                            alternative="greater")[1] < 0.05:
                    compare = "↑"
                    wdl[0] += 1
                elif ranksums(source_1[tar_domain][tar_dim][tar_idx], source_2[tar_domain][tar_dim][tar_idx],
                              alternative="less")[1] < 0.05:
                    compare = "↓"
                    wdl[2] += 1
                else:
                    compare = "→"
                    wdl[1] += 1
                print(
                    f"{tar_domain}_{tar_dim}_{tar_idx}\t{np.mean(source_1[tar_domain][tar_dim][tar_idx])}\t{compare}"
                    f"\t{np.mean(source_2[tar_domain][tar_dim][tar_idx])}"
                )
                if np.mean(source_1[tar_domain][tar_dim][tar_idx]) > np.mean(source_2[tar_domain][tar_dim][tar_idx]):
                    large_mean += 1
    print(f"↑{wdl[0]}\t→{wdl[1]}\t↓{wdl[2]}", large_mean)
    return wdl


def analysis_first_pop(results_1, results_2):
    wdl = [0, 0, 0]
    source_1, source_2 = {}, {}
    tar_type = ""
    for key in results_1.keys():
        config = results_1[key]["config"]
        if len(config) == 11:
            (_, _, _, _, tar_type, tar_domain, tar_dim, tar_idx,
             sample_num, _, sample_idx) = config
        else:
            (tar_type, tar_domain, tar_dim, tar_idx, sample_idx) = config
        top_best_y_1 = [np.mean(item["pop_history"][0]) for item in results_1[key]["metric"]]
        top_best_y_2 = [np.mean(item["pop_history"][0]) for item in results_2[key]["metric"]]
        if tar_domain not in source_1.keys():
            source_1[tar_domain] = {}
        if tar_dim not in source_1[tar_domain].keys():
            source_1[tar_domain][tar_dim] = {}
        if tar_idx not in source_1[tar_domain][tar_dim].keys():
            source_1[tar_domain][tar_dim][tar_idx] = []
        if tar_domain not in source_2.keys():
            source_2[tar_domain] = {}
        if tar_dim not in source_2[tar_domain].keys():
            source_2[tar_domain][tar_dim] = {}
        if tar_idx not in source_2[tar_domain][tar_dim].keys():
            source_2[tar_domain][tar_dim][tar_idx] = []
        source_1[tar_domain][tar_dim][tar_idx] += top_best_y_1
        source_2[tar_domain][tar_dim][tar_idx] += top_best_y_2
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    large_mean = 0
    for tar_domain in tar_domains.keys():
        for tar_dim in problem_settings["valid_dims"]:
            for tar_idx in range(tar_idx_num):
                if ranksums(source_1[tar_domain][tar_dim][tar_idx], source_2[tar_domain][tar_dim][tar_idx],
                            alternative="greater")[1] < 0.05:
                    compare = "↑"
                    wdl[0] += 1
                elif ranksums(source_1[tar_domain][tar_dim][tar_idx], source_2[tar_domain][tar_dim][tar_idx],
                              alternative="less")[1] < 0.05:
                    compare = "↓"
                    wdl[2] += 1
                else:
                    compare = "→"
                    wdl[1] += 1
                print(
                    f"{tar_domain}_{tar_dim}_{tar_idx}\t{np.mean(source_1[tar_domain][tar_dim][tar_idx])}\t{compare}"
                    f"\t{np.mean(source_2[tar_domain][tar_dim][tar_idx])}"
                )
                if np.mean(source_1[tar_domain][tar_dim][tar_idx]) > np.mean(source_2[tar_domain][tar_dim][tar_idx]):
                    large_mean += 1
    print(f"↑{wdl[0]}\t→{wdl[1]}\t↓{wdl[2]}", large_mean)
    return wdl


def analysis_step_num_multi_compare(results_compares, results, step_num: int = 800):
    wdl = [[0, 0, 0] for _ in results_compares]
    source = {}
    source_compares = [{} for _ in results_compares]
    tar_type = ""
    for key in results.keys():
        config = results[key]["config"]
        if len(config) == 11:
            (_, _, _, _, tar_type, tar_domain, tar_dim, tar_idx,
             sample_num, _, sample_idx) = config
        else:
            (tar_type, tar_domain, tar_dim, tar_idx, sample_idx) = config
        top_best_y = [np.max(item["step_history"][:step_num]) for item in results[key]["metric"]]
        top_best_y_compares = [[np.max(item["step_history"][:step_num]) for item in results_compare[key]["metric"]] for
                               results_compare in results_compares]
        if tar_domain not in source.keys():
            source[tar_domain] = {}
        if tar_dim not in source[tar_domain].keys():
            source[tar_domain][tar_dim] = {}
        if tar_idx not in source[tar_domain][tar_dim].keys():
            source[tar_domain][tar_dim][tar_idx] = []
        source[tar_domain][tar_dim][tar_idx] += top_best_y
        for source_idx in range(len(source_compares)):
            if tar_domain not in source_compares[source_idx].keys():
                source_compares[source_idx][tar_domain] = {}
            if tar_dim not in source_compares[source_idx][tar_domain].keys():
                source_compares[source_idx][tar_domain][tar_dim] = {}
            if tar_idx not in source_compares[source_idx][tar_domain][tar_dim].keys():
                source_compares[source_idx][tar_domain][tar_dim][tar_idx] = []
            source_compares[source_idx][tar_domain][tar_dim][tar_idx] += top_best_y_compares[source_idx]
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    large_mean = [0 for _ in results_compares]
    all_means = [[] for _ in range(len(source_compares) + 1)]
    for tar_domain in tar_domains.keys():
        for tar_dim in problem_settings["valid_dims"]:
            for tar_idx in range(tar_idx_num):
                for source_idx in range(len(source_compares)):
                    if ranksums(source_compares[source_idx][tar_domain][tar_dim][tar_idx],
                                source[tar_domain][tar_dim][tar_idx],
                                alternative="greater")[1] < 0.05:
                        compare = "↑"
                        wdl[source_idx][0] += 1
                    elif ranksums(source_compares[source_idx][tar_domain][tar_dim][tar_idx],
                                  source[tar_domain][tar_dim][tar_idx],
                                  alternative="less")[1] < 0.05:
                        compare = "↓"
                        wdl[source_idx][2] += 1
                    else:
                        compare = "→"
                        wdl[source_idx][1] += 1
                    if np.mean(source[tar_domain][tar_dim][tar_idx]) < np.mean(
                            source_compares[source_idx][tar_domain][tar_dim][tar_idx]):
                        large_mean[source_idx] += 1
                    all_means[source_idx].append(np.mean(source_compares[source_idx][tar_domain][tar_dim][tar_idx]))
                all_means[-1].append(np.mean(source[tar_domain][tar_dim][tar_idx]))
    print(np.mean(all_means[-1]))
    for source_idx in range(len(source_compares)):
        print(f"↑{wdl[source_idx][0]}\t→{wdl[source_idx][1]}\t↓{wdl[source_idx][2]}", large_mean[source_idx],
              np.mean(all_means[source_idx]))
    return wdl, large_mean


def analysis_step_num_multi_compare_with_group(results_compares, results, step_num: int = 800):
    wdl = [{} for _ in results_compares]
    source = {}
    source_compares = [{} for _ in results_compares]
    tar_type = ""
    for key in results.keys():
        config = results[key]["config"]
        if len(config) == 11:
            (_, _, _, _, tar_type, tar_domain, tar_dim, tar_idx,
             sample_num, _, sample_idx) = config
        else:
            (tar_type, tar_domain, tar_dim, tar_idx, sample_idx) = config
        top_best_y = [np.max(item["step_history"][:step_num]) for item in results[key]["metric"]]
        top_best_y_compares = [[np.max(item["step_history"][:step_num]) for item in results_compare[key]["metric"]] for
                               results_compare in results_compares]
        if tar_domain not in source.keys():
            source[tar_domain] = {}
        if tar_dim not in source[tar_domain].keys():
            source[tar_domain][tar_dim] = {}
        if tar_idx not in source[tar_domain][tar_dim].keys():
            source[tar_domain][tar_dim][tar_idx] = []
        source[tar_domain][tar_dim][tar_idx] += top_best_y
        for source_idx in range(len(source_compares)):
            if tar_domain not in source_compares[source_idx].keys():
                source_compares[source_idx][tar_domain] = {}
            if tar_dim not in source_compares[source_idx][tar_domain].keys():
                source_compares[source_idx][tar_domain][tar_dim] = {}
            if tar_idx not in source_compares[source_idx][tar_domain][tar_dim].keys():
                source_compares[source_idx][tar_domain][tar_dim][tar_idx] = []
            source_compares[source_idx][tar_domain][tar_dim][tar_idx] += top_best_y_compares[source_idx]
    tar_idx_num = problem_settings["training_gate_ins_num"] if tar_type == "gate_train" else problem_settings[
        "valid_ins_num"]
    tar_domains = train_problem_domains if tar_type == "gate_train" else valid_problem_domains
    large_mean = [{} for _ in results_compares]
    all_means = [[] for _ in range(len(source_compares) + 1)]
    for tar_domain in tar_domains.keys():
        for tar_dim in problem_settings["valid_dims"]:
            for tar_idx in range(tar_idx_num):
                for source_idx in range(len(source_compares)):
                    if tar_domain not in wdl[source_idx].keys():
                        wdl[source_idx][tar_domain] = [0, 0, 0]
                        large_mean[source_idx][tar_domain] = 0
                    if ranksums(source_compares[source_idx][tar_domain][tar_dim][tar_idx],
                                source[tar_domain][tar_dim][tar_idx],
                                alternative="greater")[1] < 0.05:
                        compare = "↑"
                        wdl[source_idx][tar_domain][0] += 1
                    elif ranksums(source_compares[source_idx][tar_domain][tar_dim][tar_idx],
                                  source[tar_domain][tar_dim][tar_idx],
                                  alternative="less")[1] < 0.05:
                        compare = "↓"
                        wdl[source_idx][tar_domain][2] += 1
                    else:
                        compare = "→"
                        wdl[source_idx][tar_domain][1] += 1
                    if np.mean(source[tar_domain][tar_dim][tar_idx]) < np.mean(
                            source_compares[source_idx][tar_domain][tar_dim][tar_idx]):
                        large_mean[source_idx][tar_domain] += 1
                    all_means[source_idx].append(np.mean(source_compares[source_idx][tar_domain][tar_dim][tar_idx]))
                all_means[-1].append(np.mean(source[tar_domain][tar_dim][tar_idx]))
    print(np.mean(all_means[-1]))
    for source_idx in range(len(source_compares)):
        for tar_domain in tar_domains.keys():
            print(
                f"{source_idx}\t{tar_domain}\t↑{wdl[source_idx][tar_domain][0]}\t→{wdl[source_idx][tar_domain][1]}\t↓{wdl[source_idx][tar_domain][2]}",
                large_mean[source_idx][tar_domain],
                np.mean(all_means[source_idx]))
    return wdl, large_mean


