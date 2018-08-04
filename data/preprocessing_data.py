import time
import numpy as np
from scipy.sparse import csr_matrix
import xmltodict
import pandas as pd
import collections
import pickle
import nltk
from nltk.tokenize import RegexpTokenizer

def preprocess_irishtimes(data, max_vocab=None, years=None):
    # https://www.kaggle.com/therohk/ireland-historical-news
    t0 = time.time()
    tokenizer = RegexpTokenizer(r'\w+')
    # data = data[data['headline_category'] == 'sport.soccer']
    # data = data[data['headline_category'] == 'business']
    not_relevant = ['at', 'as', 'an', 'of', 'from', 'to', 'on', 'in', 'with',
                    'for', 'by', 'about', 'into', 'the', 'and']
    clean_data = list()
    # print('Total number of docs: {}'.format(len(data['headline_text'])))
    full_vocab = dict()
    for i, headline in enumerate(data['headline_text']):
        try:
            tknd_hdln = tokenizer.tokenize(headline)
            # remove single letters, articles and preprositions
            tknd_hdln = [word.lower() for word in tknd_hdln
                         if (len(word) > 1) and word.lower() not in not_relevant]
            # vocab = vocab.union(set(tknd_hdln))  # way slower than line below!!
            # vocab |= set(tknd_hdln)
            for word in tknd_hdln:
                if word not in full_vocab:
                    full_vocab[word] = 1
                else:
                    full_vocab[word] += 1
            clean_data.append(tknd_hdln)
        except TypeError:
            print('Problem reading headline. Content: {}'.format(headline))

    if max_vocab is None:
        vocab = full_vocab
    else:
        vocab = {} # list()
        for i, w in enumerate(sorted(full_vocab, key=full_vocab.get, reverse=True)):
            vocab[w] = i
            if len(vocab) >= max_vocab:
                break
    # construct document-term matrix
    dtm = np.zeros((len(clean_data), len(vocab)), dtype=np.uint8)
    for i, headline in enumerate(clean_data):
        for j, w in enumerate(headline):
            if w in vocab.keys():
                dtm[i, vocab[w]] += 1
    dtm_sp = csr_matrix(dtm)
    with open('../data/irishtimes_processed.pkl', 'wb') as fh:
        pickle.dump([dtm_sp, vocab], fh)

    print('Final vocabulary size: {}'.format(len(vocab)))
    print('Preprocessing time: {} secs'.format(time.time()-t0))


def preprocess_blogs(posts_path, max_vocab=None):
    # http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
    import glob
    import xml.etree.ElementTree as ET

    # txtfiles = []
    # for file in glob.glob(path+"*.xml"):
    # file = posts_path + '4332352.male.26.Communications-Media.Libra.xml'
    # tree = ET.parse(file)
    # root = tree.getroot()
    # print(root)
    for file in glob.glob(posts_path+"*.xml"):
        with open(file) as fd:
            try:
                doc = xmltodict.parse(fd.read())
            except:
                print(file)
            print(len(doc['Blog']['post']))



if __name__ == '__main__':
    data = pd.read_csv('../data/irishtimes-date-text.csv')
    preprocess_irishtimes(data, max_vocab=1000)

    # posts_path = '../data/blogs/'
    # preprocess_blogs(posts_path, max_vocab=1000)
