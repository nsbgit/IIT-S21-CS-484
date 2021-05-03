#import pandas as pd
from sklearn import metrics

pred_prob = [0.8234,0.1766,0.9953,0.0047,0.8038,0.1962,0.6073,0.3927] 
class_tree =['LOSS','WIN','LOSS','WIN','LOSS','WIN','WIN','LOSS']

AUC = metrics.roc_auc_score(class_tree, pred_prob)       
print('AUC = ', AUC)

# [0.8234,0.9953,0.8038,0.6073]