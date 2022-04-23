from sklearn.model_selection import train_test_split
from pathlib import Path
import tensorflow as tf
from config import *
import numpy as np
import logging
import random
import os

def random_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed) # Python general
    np.random.seed(seed)
    random.seed(seed) # Python random
    tf.random.set_seed(seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'

def get_data(data_dir):
    p = Path(data_dir)
    inputImgPaths = sorted(list(p.glob('**/*.png')))
    inputLabelPaths = [CLASS_MAPPING[str(img).split(os.path.sep)[-2]] for img in inputImgPaths]
    X_train, X_test, y_train, y_test = train_test_split(inputImgPaths, inputLabelPaths, test_size=TEST_SPLIT, random_state=SEED)
    train_counts = np.unique(y_train, return_counts=True)
    test_counts = np.unique(y_test, return_counts=True)
    logging.info(f"We have {train_counts[1][0]} samples of CAP, {train_counts[1][1]} samples of COVID, {train_counts[1][2]} samples of NonCOVID in Training dataset")
    logging.info(f"We have {test_counts[1][0]} samples of CAP, {test_counts[1][1]} samples of COVID, {test_counts[1][2]} samples of NonCOVID in Training dataset")
    return (X_train, X_test, y_train, y_test)

class LoggerWriter:
    def __init__(self, logfct):
        self.logfct = logfct
        self.buf = []

    def write(self, msg):
        if msg.endswith('\n'):
            self.buf.append(msg.rstrip('\n'))
            self.logfct(''.join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self):
        pass