# import pandas as pd
import pickle
# from utils import preprocess_irishtimes


if __name__ == '__main__':

    with open('../data/irishtimes_processed.pkl', 'rb') as fh:
        dtm, vocab = pickle.load(fh)
    print(dtm.shape)
