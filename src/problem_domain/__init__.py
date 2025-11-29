from src.types_ import *

from src.problem_domain.base_problem import BaseProblem
from src.problem_domain.com_influence_max_problem import ComInfluenceMaxProblem
from src.problem_domain.compiler_args_selection_problem import CompilerArgsSelectionProblem
from src.problem_domain.contamination_problem import ContaminationProblem
from src.problem_domain.surrogate_problem import SurrogateProblem
from src.problem_domain.match_max_problem import MatchMaxProblem
from src.problem_domain.zero_one_knapsack_problem import ZeroOneKnapsackProblem
from src.problem_domain.max_cut_problem import MaxCutProblem

train_problem_domains: Dict[str, Type[BaseProblem]] = {
    "match_max_problem": MatchMaxProblem,
    "max_cut_problem": MaxCutProblem,
    "zero_one_knapsack_problem": ZeroOneKnapsackProblem,
}
valid_problem_domains: Dict[str, Type[BaseProblem]] = {
    "match_max_problem": MatchMaxProblem,
    "max_cut_problem": MaxCutProblem,
    "zero_one_knapsack_problem": ZeroOneKnapsackProblem,
    "contamination_problem": ContaminationProblem,
    "com_influence_max_problem": ComInfluenceMaxProblem,
    "compiler_args_selection_problem": CompilerArgsSelectionProblem,
}
problem_cost: Dict[str, int] = {
    "match_max_problem": 1,
    "max_cut_problem": 1,
    "zero_one_knapsack_problem": 1,
    "contamination_problem": 1,
    "com_influence_max_problem": 4,
    "compiler_args_selection_problem": 1,
}
problem_settings: Dict[str, Union[int, List[int]]] = {
    "training_dims": [30, 35, 40],
    "valid_dims": [40, 60, 80, 100],
    "training_ins_num": 3,
    "training_gate_ins_num": 3,
    "valid_ins_num": 3,
}
