from sklearn.metrics import f1_score, recall_score


def f1(y_true, y_pred):
    return round(f1_score(y_true, y_pred, average='micro') * 100, 3)


def uar(y_true, y_pred):
    return round(recall_score(y_true, y_pred, average='macro') * 100, 3)


def eval_metric(predicts, targets, partition_name):
    results = {'f1': f1(targets, predicts), 'uar': uar(targets, predicts)}
    results['combine'] = round((0.66 * results['f1'] + 0.34 * results['uar']), 3)
    print(f'Results in {partition_name}:\n')
    print("  - f1: ", results['f1'])
    print("  - uar: ", results['uar'])
    print("  - combined:", results['combine'])
