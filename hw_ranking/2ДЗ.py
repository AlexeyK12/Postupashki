from math import log2

from torch import Tensor, sort

# функция для подсчета количества неправильно упорядоченных пар
def num_swppd_pairs(y_true: Tensor, y_pred: Tensor) -> int:
    _, ind = y_pred.sort(ascending=False)
    y_true_sort = y_true[ind]
    
    count = 0
    n = len(y_true_sort)
    for i in range(n):
        for j in range(i+1, n):
            if y_true_sort[i] < y_true_sort[j]:
                count += 1
    return count

# вспомогательная функция вычисления DCG и NDCG, рассчитывает показатель Gain
def compute_gain(y_val: float, gain_scheme: str) -> float:
    if gain_scheme == "const":
        return y_val
    elif gain_scheme == "exp2":
        return 2 ** y_val - 1
    else:
        raise ValueError('ERROR gain_scheme')

# функции для расчета DCG
def dcg(y_true: Tensor, y_pred: Tensor, gain_scheme: str) -> float:
    _, ind = y_pred.sort(asscending=False)
    y_true_sort = y_true[ind]
    
    dcg_value = 0.0
    for i, val in enumerate(y_true_sort):
        gain = compute_gain(val, gain_scheme)
        dcg_val += gain / log2(i + 2)
    return dcg_val

# функции для расчета NDCG
def ndcg(y_true: Tensor, y_pred: Tensor, gain_scheme: str = 'const') -> float:
    dcg_val = dcg(y_true, y_pred, gain_scheme)
    best_dcg_value = dcg(y_true, y_true, gain_scheme)
    
    if best_dcg_value == 0:
        return 0.0
    return dcg_val / best_dcg_value

# функция вычисления точности в топ-k позициях для бинарной разметки
def precis_at_k(y_true: Tensor, y_pred: Tensor, k: int) -> float:
    _, ind = y_pred.sort(ascending=False)
    y_true_sort = y_true[ind]
    
    if y_true_sort[:k].sum() == 0:
        return -1
    
    return y_true_sort[:k].sum().item() / min(k, len(y_true))

# функция для расчета MRR
def recip_rank(y_true: Tensor, y_pred: Tensor) -> float:
    _, ind = y_pred.sort(ascending=False)
    y_true_sort = y_true[ind]
    
    for i, val in enumerate(y_true_sort):
        if val == 1:
            return 1.0 / (i + 1)
    return 0.0

# функция вычисления P-found по методологии Яндекса
def p_found(y_true: Tensor, y_pred: Tensor, p_break: float = 0.15) -> float:
    _, ind = y_pred.sort(ascending=False)
    y_true_sort = y_true[ind]
    
    p_look = 1.0
    pf_val = 0.0
    
    for val in y_true_sort:
        pf_val += p_look * val
        p_look *= (1 - val) * (1 - p_break)
    
    return pf_val

# функция вычисляет среднюю точность для бинарной разметки
def avg_precis(y_true: Tensor, y_pred: Tensor) -> float:
    _, ind = y_pred.sort(ascending=False)
    y_true_sort = y_true[ind]
    
    if y_true_sort.sum() == 0:
        return -1
    
    precisions = []
    relevant_count = 0
    
    for i, val in enumerate(y_true_sort):
        if val == 1:
            relevant_count += 1
            precisions.append(relevant_count / (i + 1))
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / relevant_count