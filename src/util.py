from src.types_ import *

def hold_best(sequence: List[Union[int, float]]) -> List[Union[int, float]]:
    new_sequence = [sequence[0]]
    for i in range(1, len(sequence)):
        new_sequence.append(max(new_sequence[i - 1], sequence[i]))
    return new_sequence
