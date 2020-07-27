import numpy as np
import pandas as pd

if __name__ == '__main__':
    topic = np.load('./fine_tune_albert/experiments/outputs/topicFalse/test_final.npy')
    df = pd.read_csv('./svms/test.csv')
    df['prediction_topic'] = topic
    df.to_csv('c2_muse_topic_test_14_2.csv', index=False)