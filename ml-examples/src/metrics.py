import numpy as np

def get_tp_tn_fp_fn(conf_matrix):

    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP 
    FN = conf_matrix.sum(axis=1) - TP
    TN = conf_matrix.sum() - (FP + FN + TP)

    return TP, TN, FP, FN

def scores_per_class(conf_matrix):

    TP, TN, FP, FN = get_tp_tn_fp_fn(conf_matrix)

    # Precision or positive predictive value
    precision = TP / (TP + FP)

    recall = TP / (TP + FN)

    f1 = 2 * (precision * recall) / (precision + recall)

    # Overall accuracy
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    return np.nan_to_num(accuracy), np.nan_to_num(f1)
