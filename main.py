import pandas as pd
from word_cloud import show_wordcloud
import lyrics_analysis

artist = 'Ed Sheeran' # Change the artist name here to get insight of a specific artist
df = pd.read_csv('D:/kaggle/songlyrics/songdata.csv')
df_filter = df[df['artist'] == artist]
df_filter['text'] = df_filter['text'].apply(lambda x: str(x).lower())
song_list = df_filter['song'].tolist()

# Lyrics list
lyrics_list = [row['text'] for i, row in df_filter.iterrows()]


# Word cloud for all songs of a particular singer
all_text = ' '.join(lyrics_list)
show_wordcloud(all_text)

# Document to TF-IDF matrix
tfidf_X = lyrics_analysis.to_tfidf_matrix(lyrics_list)

# Descriptive statistics songs from an artist
lyrics_length_list = [len(x.split(' ')) for x in lyrics_list]
mean_length = sum(lyrics_length_list)/len(lyrics_length_list)
min_length = min(lyrics_length_list)
max_length = max(lyrics_length_list)

print('Average length of songs: {} words'.format(round(mean_length, 0)))
print('Minimum length of a sonh: {} words'.format(min_length))
print('Maximum length of a song: {} words'.format(max_length))

# Calculate cosine similarity
lyrics_analysis.calc_cosine_sim(tfidf_X, song_list)

# Trigram analysis
lyrics_analysis.analyse_ngrams(ngram=3, lyrics_list=lyrics_list)

# Most frequent words
lyrics_analysis.most_frequent_words(lyrics_list)