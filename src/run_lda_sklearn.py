import pickle
import time
from sklearn.decomposition import LatentDirichletAllocation


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


if __name__ == '__main__':

    with open('../data/blogs_processed.pkl', 'rb') as fh:
        dtm, vocab = pickle.load(fh)
    dtm_dense = dtm.todense()
    tf_feature_names = list(vocab.keys())

    n_samples = 2000
    n_features = 1000
    n_components = 5
    n_top_words = 20

    lda = LatentDirichletAllocation(n_components=n_components, max_iter=10,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    t0 = time.time()
    lda.fit(dtm_dense)
    print("done in %0.3fs." % (time.time() - t0))

    print("\nTopics in LDA model:")
    # tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
