from src.baseline_init.kaes_init import KAESInitializer
from src.baseline_init.svm_ss import SVMSSInitializer
from src.network.decoder_mapping import DecoderMapping
from src.problem_domain import BaseProblem
from src.types_ import *

def eval_MPI_time():
    start = time.time()
    dp = DecoderMapping(
        ins_dir=Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance"),
        surrogate_dir=Path(os.path.dirname(os.path.realpath(__file__)), "../../out/surrogate_models"),
        map_dir=Path(os.path.dirname(os.path.realpath(__file__)), "../../temp"),
        src_domain="match_max_problem", tar_domain="compiler_args_selection_problem",
        src_type="train", tar_type="test",
        src_dim=40, tar_dim=40, src_idx=0, tar_idx=0,
        sample_num=64, src_coefficient=4, sample_idx=0
    )
    dp.get_correlation()
    print("USE Time: ", time.time() - start)
    dp.fine_tuning_mapping_decoder()
    print("USE Time: ", time.time() - start)
    dp.get_topk_target_solution(k=4)
    print("USE Time: ", time.time() - start)


def eval_SVM_SS_time():
    start = time.time()
    ins_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    tar_type, tar_domain = "test", "match_max_problem"
    tar_dim, tar_idx = 40, 0
    problem_path = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    svmss = SVMSSInitializer(pop_size=20, dim=tar_dim, eval_func=problem_instance.evaluate, max_eval=132)
    svmss.initialize()
    print("SVM-SS USE Time: ", time.time() - start)


def eval_KAES_time():
    start = time.time()
    ins_dir = Path(os.path.dirname(os.path.realpath(__file__)), "../../data/problem_instance")
    tar_type, tar_domain = "test", "match_max_problem"
    tar_dim, tar_idx = 40, 0
    problem_path = Path(ins_dir, tar_type, f"{tar_domain}_{tar_dim}_{tar_idx}")
    problem_instance: BaseProblem = pickle.load(open(Path(problem_path, "problem.pkl"), 'rb'))
    kaes = KAESInitializer(pop_size=20, dim=tar_dim, eval_func=problem_instance.evaluate, ins_dir=ins_dir,
                           max_eval=132)
    kaes.initialize()
    print("KAES USE Time: ", time.time() - start)


if __name__ == '__main__':
    eval_SVM_SS_time()
    eval_KAES_time()
    eval_MPI_time()
