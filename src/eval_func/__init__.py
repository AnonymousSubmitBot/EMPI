from src.eval_func.cpu_eval import eval_problem_solutions, eval_problem_EA, eval_problem_EA_detail
from src.eval_func.cpu_eval import generate_svmss_init_solution, generate_obl_init_solution, generate_qi_init_solution
from src.eval_func.gpu_eval import fit_surrogate, map_decoder, generate_init_solution, calculate_correlation

eval_funcs = {
    "eval_problem_solutions": eval_problem_solutions,
    "eval_surrogate_training": fit_surrogate,
    "eval_surrogate_mapping": map_decoder,
    "generate_init_solution": generate_init_solution,
    "generate_svmss_init_solution": generate_svmss_init_solution,
    "generate_obl_init_solution": generate_obl_init_solution,
    "generate_qi_init_solution": generate_qi_init_solution,
    "calculate_correlation": calculate_correlation,
    "eval_problem_EA": eval_problem_EA,
    "eval_problem_EA_detail": eval_problem_EA_detail
}
