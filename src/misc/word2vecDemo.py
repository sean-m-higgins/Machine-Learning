# import gensim.downloader as api
# wv = api.load('word2vec-google-news-300')

# for i, word in enumerate(wv.vocab):
#     if i == 10:
#         break
#     print(word)

# vec_king = wv['king']

# try:
#     vec_cameroon = wv['cameroon']
# except KeyError:
#     print("The word 'cameroon' does not appear in this model")

# pairs = [
#     ('car', 'minivan'),   # a minivan is a kind of car
#     ('car', 'bicycle'),   # still a wheeled vehicle
#     ('car', 'airplane'),  # ok, no wheels, but still a vehicle
#     ('car', 'cereal'),    # ... and so on
#     ('car', 'communism'),
# ]
# for w1, w2 in pairs:
#     print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))


# print(wv.most_similar(positive=['car', 'minivan'], topn=5))

# print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))



# =============================================================================
# EXAMPLE FROM -- https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
import pandas as pd
import numpy as np
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
from gensim.models import Word2Vec
# https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use("TkAgg")
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

cars_df = pd.read_csv("./data/cars/data.csv")
print(cars_df.head())

# preprocessing
cars_df['Maker_Model']= cars_df['Make']+ " " + cars_df['Model']

# Select features from original dataset to form a new dataframe
df1 = cars_df[['Engine Fuel Type','Transmission Type','Driven_Wheels','Market Category','Vehicle Size', 'Vehicle Style', 'Maker_Model']]

# For each row, combine all the columns into one column
df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)

# Store them in a pandas dataframe
df_clean = pd.DataFrame({'clean': df2})

# Create the list of list format of the custom corpus for gensim modeling 
sent = [row.split(',') for row in df_clean['clean']]

# show the example of list of list format of the custom corpus for gensim modeling 
print(sent[:2])

# We can train the genism word2vec model with our own custom corpus as following:
# sg: The training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW.
model = Word2Vec(sent, min_count=1,size= 50,workers=3, window =3, sg = 1)

print(model['Toyota Camry'])
print(model['BMW 1 Series M'])

print(model.similarity('Porsche 718 Cayman', 'Nissan Van'))
print(model.similarity('Porsche 718 Cayman', 'Mercedes-Benz SLK-Class'))
print(model.most_similar('Mercedes-Benz SLK-Class')[:5])


def cosine_distance (model, word,target_list , num) :
    cosine_dict ={}
    word_list = []
    a = model[word]
    for item in target_list :
        if item != word :
            b = model [item]
            cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            cosine_dict[item] = cos_sim
    dist_sort=sorted(cosine_dict.items(), key=lambda dist: dist[1],reverse = True) ## in Descedning order 
    for item in dist_sort:
        word_list.append((item[0], item[1]))
    return word_list[0:num]

# only get the unique Maker_Models
Maker_Model = list(cars_df.Maker_Model.unique()) 

# Show the most similar Mercedes-Benz SLK-Class by cosine distance
cd = cosine_distance (model,'Mercedes-Benz SLK-Class',Maker_Model,5)
print(cd)

def display_closestwords_tsnescatterplot(model, word, size):
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]

    close_words = model.similar_by_word(word)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

display_closestwords_tsnescatterplot(model, 'Porsche 718 Cayman', 50) 
