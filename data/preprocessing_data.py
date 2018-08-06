import time
import numpy as np
from scipy.sparse import csr_matrix
import xmltodict
import pandas as pd
import collections
import pickle
import codecs
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def process_data(docs, max_vocab=None):
    tokenizer = RegexpTokenizer(r'\w+')
    # data = data[data['headline_category'] == 'sport.soccer']
    # data = data[data['headline_category'] == 'business']
    stop_words = ("a", "about", "above", "after", "again",
    "against", "all", "am", "an", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each",
    "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
    "should", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
     "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's",
     "when", "when's", "where", "where's", "which", "while", "who", "who's",
     "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
     "you're", "you've", "your", "yours", "yourself", "yourselves", "urllink",
     "one", "http", "www", "like", "got", "get", "mr")
    stop_words = set(stop_words)

    stop_words_nltk = set(stopwords.words('english'))
    stop_words = stop_words.union(stop_words_nltk)
    print(len(stop_words))
    clean_data = list()
    # print('Total number of docs: {}'.format(len(data['headline_text'])))
    full_vocab = dict()
    for i, headline in enumerate(docs):
        try:
            tknd_hdln = tokenizer.tokenize(headline)
            # remove single letters, articles and preprositions
            tknd_hdln = [word.lower() for word in tknd_hdln
                         if (len(word) > 1) and word.lower() not in stop_words]
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
            print('{}: {}'.format(i, w))
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
    return dtm_sp, vocab


def preprocess_irishtimes(data, max_vocab=None, years=None):
    # https://www.kaggle.com/therohk/ireland-historical-news
    t0 = time.time()
    dtm_sp, vocab = process_data(data['headline_text'], max_vocab=max_vocab)

    with open('../data/irishtimes_processed.pkl', 'wb') as fh:
        pickle.dump([dtm_sp, vocab], fh)

    print('Final vocabulary size: {}'.format(len(vocab)))
    print('Preprocessing time: {} secs'.format(time.time()-t0))


def preprocess_blogs(posts_path, max_vocab=None):
    # http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
    import glob
    t0 = time.time()
    all_posts = list()
    for file in glob.glob(posts_path+"*.xml"):
        with codecs.open(file, encoding='utf-8', errors='ignore') as fd:
            str_xml = fd.read().replace('&', ' and ').replace('<>', '')
            doc = xmltodict.parse(str_xml)
            if isinstance(doc['Blog']['post'], list):  # for multiple posts
                for post in doc['Blog']['post']:
                    all_posts.append(post)
            else:
                all_posts.append(doc['Blog']['post'])
                # print('{} ({}): {}'.format(file, type(doc['Blog']['post']), len(doc['Blog']['post'])))
    print('total number of blogs: {}'.format(len(all_posts)))
    dtm_sp, vocab = process_data(all_posts, max_vocab=max_vocab)

    with open('../data/blogs_processed.pkl', 'wb') as fh:
        pickle.dump([dtm_sp, vocab], fh)

    print('Final vocabulary size: {}'.format(len(vocab)))
    print('Preprocessing time: {} secs'.format(time.time()-t0))



if __name__ == '__main__':
    data = pd.read_csv('../data/irishtimes-date-text.csv')
    preprocess_irishtimes(data, max_vocab=1000)

    # posts_path = '../data/short_blogs/'
    # preprocess_blogs(posts_path, max_vocab=1000)
