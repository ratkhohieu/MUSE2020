import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import time
import sys


def pad_if_need(paths, size):
    diff = len(paths) - size
    if diff < 0:
        if abs(diff) > len(paths):  # lon hon nhieu hon 2 lan so voi paths nen co the lap lai path
            up_sampling = paths[np.random.choice(paths.shape[0], abs(diff), replace=True)]
        else:  # lon hon khong qua 2 lan so voi paths nen de khong lap lai path
            up_sampling = paths[np.random.choice(paths.shape[0], abs(diff), replace=False)]
        paths = np.concatenate([paths, up_sampling])
    return paths


def pooling_segment(video_features: np.array, stride=2, pool_out=16):
    if len(video_features) < pool_out:
        video_features = pad_if_need(video_features, pool_out)
        return [video_features]
    else:
        ret = []
        i = 0

        while i < len(video_features) - pool_out:
            ret.append(video_features[i:i + pool_out])
            i += stride
        return ret


def pooling_segments(feature_df: pd.DataFrame, label_df: pd.DataFrame, stride=2, pool_out=16):
    col_features = [col for col in feature_df.columns if not col in ['timestamp', 'segment_id']]
    all_segments = feature_df['segment_id'].unique()
    all_segment_features = []
    all_segment_labels = []
    all_segment_ids = []
    for segment in all_segments:
        segment_df = feature_df[feature_df['segment_id'] == segment]
        segment_features = segment_df[col_features].values
        label = label_df[label_df['segment_id'] == segment]['class_id'].values
        pooled_feature = pooling_segment(segment_features, stride, pool_out)
        pooled_label = [label] * len(pooled_feature)
        segment = [segment] * len(pooled_feature)
        all_segment_features += pooled_feature
        all_segment_labels += pooled_label
        all_segment_ids += segment
    all_segment_features = np.asarray(all_segment_features)
    all_segment_labels = np.asarray(all_segment_labels)
    all_segment_ids = np.asarray(all_segment_ids)
    return all_segment_features, all_segment_labels, all_segment_ids


def prepare_data(root_feature_folder, root_label_folder, label, partition_info, feature_set, mode=None):
    train_lab, train_feat, train_id, devel_lab, devel_feat, devel_id, test_lab, test_feat, test_id = [], [], [], [], [], [], [], [], []
    partition_info = pd.read_csv(partition_info)
    feature_folder = root_feature_folder
    label_folder = os.path.join(root_label_folder, label)

    print('\n ' + feature_set + ': ' + label)

    print('\n Preparing Partitions')
    for index, row in tqdm(partition_info.iterrows(), total=partition_info.shape[0]):
        filename_id = str(row['Id']) + '.csv'
        row_partition = row['Proposal']

        label_df = pd.read_csv(os.path.join(label_folder, filename_id), index_col=None, dtype=np.float64)
        feature_df = pd.read_csv(os.path.join(feature_folder, feature_set, filename_id),
                                 index_col=None, dtype=np.float64)

        features, labels, id_segment = pooling_segments(feature_df, label_df)

        if (row_partition == 'train') & (mode == 'train'):
            train_lab.append(labels)
            train_feat.append(features)
            train_id.append(id_segment)
        if (row_partition == 'devel') & (mode == 'devel'):
            devel_lab.append(labels)
            devel_feat.append(features)
            devel_id.append(id_segment)
        if (row_partition == 'test') & (mode == 'test'):
            test_lab.append(labels)
            test_feat.append(features)
            test_id.append(id_segment)

        list_value = [train_lab, train_feat, train_id, devel_lab, devel_feat, devel_id, test_lab, test_feat, test_id]
        list_key = ['train_lab', 'train_feat', 'train_id', 'devel_lab', 'devel_feat', 'devel_id', 'test_lab',
                    'test_feat', 'test_id']
        dict_feature = dict(zip(list_key, list_value))
    for key, item in dict_feature.items():
        item = np.asarray(item)
        if len(item)!=0:
            dict_feature[key] = np.concatenate(item, axis=0)
    return dict_feature


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
