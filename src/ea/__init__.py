from typing import Dict, Type, Any

from src.ea.brkga import BRKGA
from src.ea.ea import EA
from src.ea.elite_ga import EliteGA
from src.ea.rpeda import RPEDA
from src.ea.umda import UMDA
from src.ea.only_init import OnlyInit

ea_algs: Dict[str, Type[EA]] = {
    "brkga": BRKGA,
    "elite_ga": EliteGA,
    "umda": UMDA,
    "rpeda": RPEDA,
    "only_init": OnlyInit,
}

ea_parameter: Dict[str, Dict[str, Any]] = {
    "brkga": {
        "max_eval": 800,
        "max_iter": 100,
        "n_elites": 16,
        "n_offsprings": 56,
        "n_mutants": 8,
        "bias": 0.7
    },
    "elite_ga": {
        "max_eval": 800,
        "max_iter": 100,
        "pop_size": 64,
        "mut_prob": 0.01,
        "elite": True,
        "elite_k": 1
    }
}
