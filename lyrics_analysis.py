from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform
from itertools import chain
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.util import ngrams
import pandas as pd
import itertools






# documents to matrix by counting occurences
def to_occurence_matrix(text_list):
    stopWords = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words = stopWords)
    X = vectorizer.fit_transform(text_list)
    X = X.toarray()
    return X

# documents to matrix using tf-idf scores
def to_tfidf_matrix(text_list):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit_transform(text_list)
    tfidf_X = tfidf_vectorizer.fit_transform(text_list)
    tfidf_X = tfidf_X.toarray()
    return tfidf_X




# Cosine similarity of songs by the same singer
def calc_cosine_sim(tfidf_X, song_list):
    similar_songs = {}
    cosine_dist = squareform(pdist(tfidf_X, 'cosine'))
    for song in cosine_dist:
        res = itertools.compress(song_list, map(lambda x: x < 0.4, song)) # Considering simlarity > 60%
        res = [_ for _ in res ]
        if len(res) > 1:
            res = tuple(res)
            similar_songs[hash(res)] = res
    return similar_songs.values()


def song_to_token(text):
    return ' '.join(text.split()).replace('\n', '').split(' ')

# TOP 20 bigrams/trigrams used by the artist
def analyse_ngrams(ngram, lyrics_list):
    trigramfdist = []
    special_chars = '"(),-.' # possible special characters can appear in a string
    for line in lyrics_list:
        line = ''.join(c for c in line if c not in special_chars)
        tokens = song_to_token(line)
        bigrams = ngrams(tokens, ngram)
        trigramfdist.append(list(bigrams))

    bigramfdist = list(chain.from_iterable(trigramfdist))
    concatenate_bigrams = [x[0] + ' ' + x[1] + ' ' + x[2] for x in bigramfdist]
    counts = pd.Series(concatenate_bigrams).value_counts()[:20]
    counts.plot.barh(width=0.9, color='red')
    plt.xlabel('Frequency')
    plt.show()

# Most used words by the specific singer in their songs
def most_frequent_words(lyrics_list):
    cachedStopWords = stopwords.words("english")
    all_words = []
    for song in lyrics_list:
        text = ' '.join([word for word in song.split() if word not in cachedStopWords])
        tokens = song_to_token(text)
        all_words.append(tokens)

    all_words = list(chain.from_iterable(all_words))
    counts = pd.Series(all_words).value_counts()[:20]
    counts.plot.barh(width=0.9, color='blue')
    plt.xlabel('Frequency')
    plt.show()

