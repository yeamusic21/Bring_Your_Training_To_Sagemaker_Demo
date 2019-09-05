###################################################################
# IMPORTS
###################################################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# Keras Imports
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, CuDNNLSTM, CuDNNGRU, Dropout, Bidirectional, Conv1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import CSVLogger
# Numpy
import numpy
numpy.random.seed(1331)
# Pandas
import pandas as pd
# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
# Visualizations
#import matplotlib.pyplot as plt
#%matplotlib inline
# Garbage Collector
import gc
import sys
# Hyperopt
#from hyperopt import fmin, tpe, hp, anneal, Trials, space_eval
# Boto is the Amazon Web Services (AWS) SDK for Python, which allows Python developers to write software that makes
# use of Amazon services like S3 and EC2. Boto provides an easy to use, object-oriented API as well as low-level direct
# service access.
import boto3
#import s3fs
# args
import argparse

###################################################################
# SAGEMAKER GATHER ARGS
###################################################################


def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    #parser.add_argument('--epochs', type=int, default=1)
    #parser.add_argument('--batch_size', type=int, default=64)
    
    #parser.add_argument('--num_words', type=int)
    #parser.add_argument('--word_index_len', type=int)
    #parser.add_argument('--labels_index_len', type=int)
    #parser.add_argument('--embedding_dim', type=int)
    #parser.add_argument('--max_sequence_len', type=int)
    
    # data directories
    parser.add_argument('--data', type=str, default=os.environ.get('SM_CHANNEL_DATA'))
    #parser.add_argument('--output', type=str, default=os.environ.get('SM_CHANNEL_OUTPUT'))
    #parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    #parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    
    # embedding directory
    #parser.add_argument('--embedding', type=str, default=os.environ.get('SM_CHANNEL_EMBEDDING'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    #parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()


def get_train_data(data_dir):
    
    j_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    x_train = j_df['comment_text']
    y_train = np.where(j_df['target']>0.50, 1, 0)

    print('x train', x_train.shape,'y train', y_train.shape)
    
    del j_df
    gc.collect()

    return x_train, y_train

def get_test_data(data_dir):
    
    x_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))

    print('x test', x_test.shape)

    return x_test

def get_embedding_matrix(data_dir, embed_size):

    EMBEDDING_FILES = [os.path.join(data_dir, 'crawl-300d-2M.vec'), os.path.join(data_dir, 'glove.840B.300d.txt')]
    #EMBEDDING_FILES = [os.path.join(data_dir, 'crawl-300d-2M.vec')]

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def load_embeddings(path):
        with open(path, encoding="utf-8") as f:
            return dict(get_coefs(*line.strip().split(' ')) for line in f)

    def build_matrix(word_index, path):
        embedding_index = load_embeddings(path)
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                pass
        return embedding_matrix

    embedding_matrix = np.concatenate([build_matrix(token.word_index, f) for f in EMBEDDING_FILES], axis=-1)
    embedding_matrix[0:max_features,]
    
    return embedding_matrix

if __name__ == "__main__":

    ###################################################################
    # LOAD TRAINING DATA
    ###################################################################
    print("loading training data...")

    args, _ = parse_args()
    
    X_train, y_train = get_train_data(args.data)

    ###################################################################
    # TOKENIZE AND PADDING
    ###################################################################
    print("building tokenizer, then tokenize training and add padding...")

    maxlen = 220
    max_features = 397709

    # create a tokenizer
    token = keras.preprocessing.text.Tokenizer(num_words=max_features)
    token.fit_on_texts(X_train)
    word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors
    X_train = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=maxlen)

    ###################################################################
    # EMBEDDING
    ###################################################################
    print("loading embeddings...")
    
    embed_size = 600

    embedding_matrix = get_embedding_matrix(args.data, embed_size)
    
    ###################################################################
    # CREATE & COMPILE MODEL
    ###################################################################
    print("compile model..")

    # Results from Hyperopt
    drpt_amt = 0.30
    lstm1_nrns = 30
    lstm2_nrns = 23
    epochs = 1
    batches = 1289

    # create model
    model = Sequential()
    model.add(Embedding(max_features, embed_size, input_length=maxlen, weights=[embedding_matrix]))
    model.add(Dropout(drpt_amt))
    model.add(Bidirectional(CuDNNLSTM(lstm1_nrns, return_sequences=True)))
    model.add(Dropout(drpt_amt))
    model.add(Bidirectional(CuDNNLSTM(lstm2_nrns)))
    model.add(Dropout(drpt_amt))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    ###################################################################
    # TRAINING
    ###################################################################
    print("run training ...")

    data_location = "/opt/ml/model/training.log"
    csv_logger = CSVLogger(data_location)

    model.fit(x=X_train, y=y_train, batch_size=batches, epochs=epochs, verbose=1, validation_split=0.10, callbacks=[csv_logger])

    ###################################################################
    # SCORE
    ###################################################################
    print("score test dataset ...")

    j_df = get_test_data(args.data)
    
    X_test = j_df['comment_text']
    X_test = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=maxlen)
    
    prediction = model.predict_proba(X_test)

    # create submission DataFrame
    submission = pd.DataFrame(j_df['id'])
    submission.head()

    submission['prediction'] = prediction
    submission.head()

    data_location = "/opt/ml/model/submission.csv"

    submission.to_csv(data_location, index=False)
