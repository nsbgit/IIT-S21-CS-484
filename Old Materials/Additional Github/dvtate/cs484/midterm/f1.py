def f1_score(precision, recall):
    numerator = precision * recall
    denominator = precision + recall
    return 2 * numerator / denominator

