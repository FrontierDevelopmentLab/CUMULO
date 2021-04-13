import numpy as np

from src.loader import get_most_frequent_label

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

def hist1D_per_class(loader, bins, nb_classes=8):
    
    histograms = {"target": np.zeros((nb_classes, len(bins) - 1)), "pred": np.zeros((nb_classes, len(bins) - 1))}
    counts = {"target": np.zeros((nb_classes, 1)), "pred": np.zeros((nb_classes, 1))}

    for properties, rois, labels, predictions in loader():

        # get values only for pixels that have valid property values, are cloudy
        mask = np.logical_and(~properties.mask, rois)
        
        # get target only for pixels with identified cloud types
        target_valid = np.logical_and(mask, labels.sum(-1) > -10)

        target = get_most_frequent_label(labels[target_valid])

        for cl in range(nb_classes):
            
            target_values = properties.data[target_valid][target == cl]
            pred_values = properties.data[mask][predictions[mask] == cl]

            counts["target"][cl] += target_values.size
            histograms["target"][cl] += np.histogram(target_values, bins=bins, density=False)[0]

            counts["pred"][cl] += pred_values.size
            histograms["pred"][cl] += np.histogram(pred_values, bins=bins, density=False)[0]
    
    return histograms, counts