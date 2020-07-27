import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier

def dataset(partition, feature_set):
    train = pd.read_csv(os.path.join('./data_csv/',partition,feature_set+'_train.csv'))
    valid = pd.read_csv(os.path.join('./data_csv/',partition,feature_set+'_devel.csv'))
    test = pd.read_csv(os.path.join('./data_csv/',partition,feature_set+'_test.csv'))
    return train, valid, test

def f1(y_true, y_pred):
    return round(f1_score(y_true, y_pred, average='micro') * 100, 3)

def uar(y_true, y_pred):
    return round(recall_score(y_true, y_pred, average='macro') * 100, 3)

def combine(y_true, y_pred):
    return round((0.66 * f1(y_true, y_pred) + 0.34 * uar(y_true, y_pred)), 3)


def method_training(partition, feature_set):
    train, valid, test= dataset(partition, feature_set)
    
    X_train, y_train = train.iloc[:,:-1], train['class_id']
    X_valid, y_valid = valid.iloc[:,:-1], valid['class_id']
    X_test = test.iloc[:,:-1]
    

    plt.figure(1, figsize=(20, 8))
    plt.clf()

    X_indices = np.arange(X_train.shape[-1])

    selector = SelectKBest(f_classif, k=int(X_train.shape[1]*20/100))
    selector.fit(X_train, y_train)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()

#     print(X_indices[selector.get_support()] )
#     plt.bar(X_indices - .25, scores, width=1,
#             label=r'Univariate score ($-Log(p_{value})$)')

#     plt.title("Univariate feature")
#     plt.xlabel('Feature number')
#     plt.yticks(())
#     plt.axis('tight')
#     plt.legend(loc='upper right')
#     plt.show()

    new = list(X_train[X_train.columns[selector.get_support()]].astype(str))

    X_train = X_train.loc[:,new]
    X_valid = X_valid.loc[:,new]
    X_test = X_test.loc[:,new]
    print('X_test shape: {}'.format(X_test.shape))
    
    print('X_train shape: {}'.format(X_train.shape))

    if partition == 'valence':
        model = RandomForestClassifier(max_depth= 7.40086325414854, random_state=1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        predv = model.predict(X_valid)
        acc = combine(y_valid, predv)
        print('Combine = {}'.format(acc)) 
    else:
        model = SVC(C = 0.053872988214651696, gamma='auto')
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predv = model.predict(X_valid)
        acc = combine(y_valid, predv)
        print('Combine = {}'.format(acc))
        
    return pred



if __name__ == "__main__":

    partition = ['arousal', 'valence']
    feature_set = ['vggface', 'xception']

    pred_arousal = method_training(partition[0], feature_set[0])
    pred_valence = method_training(partition[1], feature_set[1])
    test = pd.read_csv(os.path.join('./data_csv/',partition[0],feature_set[0]+'_test.csv'))
    pred_file_name = 'test.csv'
    print('Writing file ' + pred_file_name + '\n')
    prediction_df = pd.DataFrame(data={'id': test['id'],
                                       'segment_id': test['segment_id'].astype(int),
                                       'prediction_arousal': pred_arousal.astype(int),
                                       'prediction_valence': pred_valence.astype(int),
                                       'prediction_topic': test['prediction_topic'].astype(int), },
                                 columns=['id', 'segment_id', 'prediction_arousal', 'prediction_valence',
                                          'prediction_topic'])
    prediction_df.to_csv(pred_file_name, index=False)