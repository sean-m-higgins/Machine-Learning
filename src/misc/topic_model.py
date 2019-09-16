import gensim
import matplotlib.pyplot as plt
import numpy as np
import spacy
from zettel_preprocessor import ZettelPreProcessor

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
# from gensim.models.wrappers import LdaMallet ?????
from gensim.corpora import Dictionary

# import os
#
# test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
# lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
# text = open(lee_train_file).read()

baseball = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/baseball"
rheingold = "/Users/SeanHiggins/ZTextMiningPy/docs/data/zettels/rheingold-examples"

process = ZettelPreProcessor()
baseball_zettels = process.get_zettels_from_directory(baseball)
rheingold_zettels = process.get_zettels_from_directory(rheingold)

one_str_zettels = ''

for zettel in baseball_zettels:
    one_str_zettels = str(one_str_zettels + zettel + " \n")
for zettel in rheingold_zettels:
    one_str_zettels = str(one_str_zettels + zettel + " \n")

print(one_str_zettels)
nlp = spacy.load('en')

my_stop_words = ['say', 'Mr', '\'s', 'be', 'said', 'says', 'saying', '|', ' ', 'title', 'note', 'summary' 'cite',
                 'tags', 'know']
for stopword in my_stop_words:
    lexeme = nlp.vocab[stopword]
    lexeme.is_stop = True

doc = nlp(one_str_zettels)

texts, article = [], []

for word in doc:
    if word.text != '\n' and not word.is_stop and not word.is_punct:
        article.append(word.lemma_)
    if word.text == '\n':
        texts.append(article)
        article = []

# combine bigrams and add to corpus
bigram = gensim.models.Phrases(texts)

texts = [bigram[line] for line in texts]

dictionary = Dictionary(texts)

# (word id, number of times word appears in document)
corpus = [dictionary.doc2bow(text) for text in texts]

# latent semantic indexing, a popular information retrieval method,
# which works by decomposing the original matrix of words to
# maintain key topics. Gensim's implementation uses an SVD.
lsi_model = LsiModel(corpus=corpus, num_topics=2, id2word=dictionary)
lsi_topics = lsi_model.show_topics(num_topics=5)
print(lsi_topics)

# hierarchical dirichlet process is an unsupervised topic model which
# determines the number of topics on its own
hdp_model = HdpModel(corpus=corpus, id2word=dictionary)
hdp_topics = hdp_model.show_topics()
print(hdp_topics)

# latent dirichlet allocation
lda_model = LdaModel(corpus=corpus, num_topics=2, id2word=dictionary)
lda_topics = lda_model.show_topics()
print(lda_topics)

lsi_topics_clean = [[word for word, prob in topic] for topic_id, topic in lsi_model.show_topics(formatted=False)]
print(lsi_topics_clean)
hdp_topics_clean = [[word for word, prob in topic] for topic_id, topic in hdp_model.show_topics(formatted=False)]
print(hdp_topics_clean)
lda_topics_clean = [[word for word, prob in topic] for topic_id, topic in lda_model.show_topics(formatted=False)]
print(lda_topics_clean)

lsi_coherence = CoherenceModel(topics=lsi_topics_clean[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()
hdp_coherence = CoherenceModel(topics=hdp_topics_clean[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()
lda_coherence = CoherenceModel(topics=lda_topics_clean[:10], texts=texts, dictionary=dictionary, window_size=10).get_coherence()


def evaluateBarGraph(coherences, indices):
    """
    Function to plot bar graph
    :param coherences: list of coherence values
    :param indices: indices to be used to mark bars.
    Length of coherences and indices should be equal.
    """
    assert len(coherences) == len(indices)
    n = len(coherences)
    x = np.arange(n)
    plt.bar(x, coherences, width=0.2, tick_label= indices, align='center')
    plt.xlabel('Models')
    plt.ylabel('Coherence Value')
    plt.show()

evaluateBarGraph([lsi_coherence, hdp_coherence, lda_coherence], ['LSI', 'HDP', 'LDA'])







