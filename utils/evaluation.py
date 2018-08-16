import numpy as np
from sklearn import metrics


# define evaluation function
# input: prediction, ground truth
# output: auc, f1 score, acc

def evaluate(y_true, y_pred):
    fp, tp, thresholds = metrics.roc_curve(y_true, y_pred)
    m1 = metrics.auc(fp, tp)
    y_pred = np.round(y_pred)
    m2 = metrics.f1_score(y_true, y_pred)
    #m3 = metrics.precision_score(y_true, y_pred)
    #m4 = metrics.recall_score(y_true, y_pred)
    m5 = metrics.accuracy_score(y_true, y_pred)
    return [m1, m2, m5]

# helper function to save results to file
def write_results(out_file, s):
    with open(out_file, 'a') as f:
        st = ','.join(map(str, s))
        f.write(st + '\n')
