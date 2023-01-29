import re
import string

import pandas as pd
import numpy as np

import nltk
from nltk import pos_tag
from nltk import TweetTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Colour map



default_stopwords = stopwords.words('english')

default_stopwords.extend(
    list(string.punctuation) + ['would', 'could', 'get',
                                'want', 'he', 'twitter']
)

default_tokeniser = TweetTokenizer()
default_lemmatiser = WordNetLemmatizer()

def init_nltk_downloads() -> None:
    """
    Download the necessary resources for nltk, such as stopwords
    """
    resources = ['corpora/stopwords', 'corpora/wordnet',
                 'taggers/averaged_perceptron_tagger']
    for resource in resources:
        # Find .zip file instead since nltk have problem unzipping files
        try:
            nltk.find(f'{resource}.zip')
        except LookupError:
            nltk.download(resource.split('/')[-1])


def load_twitter_csv(data_file: str) -> pd.DataFrame:
    """
    Load in a csv file produce by Twitter scraper, return cleaned DataFrame
    """
    df = pd.read_csv(
        data_file,
        index_col=0,
        usecols=['conversation_id', 'created_at', 'user_id', 'tweet',
                 'language']
    )
    # Only consider English tweets, ignore neutral language
    df.query('language == "en"', inplace=True)
    df.drop(columns=['language'], inplace=True)
    df.dropna(subset=['tweet'], inplace=True)

    # Clean tweet texts
    df['tweet'] = \
        df['tweet'].apply(
            lambda txt: re.sub(
                r"(@[A-Za-z0-9_]+)|"        # Match mentions
                r"^http.+?|(\w+:\/\S+)",    # Match urls
                '',
                txt
            ).lower()
        )
    return df


# Remove stopwords and turn word into lemmatised form
# Remove stopwords and turn word into lemmatised form
def furnish(
        text: str,
        tokeniser=default_tokeniser,
        lemmatiser=default_lemmatiser,
        stop_words: list = default_stopwords
):
    final_text = []
    for word, tag in pos_tag(tokeniser.tokenize(text)):
        # Tag word as verb, nouns, etc, improves lemmatiser accuracy
        tag = tag.lower()[0]
        tag = tag if tag in ['a', 'r', 'n', 'v'] else None
        if tag:
            word = lemmatiser.lemmatize(word, tag)
        else:
            word = lemmatiser.lemmatize(word)
        if word not in stop_words:
            final_text.append(word)
    return ' '.join(final_text)



def gen_n_grams(
        df: pd.DataFrame, new_col_name="tweet_n_gram", min_len=1, max_len=3
):
    df[new_col_name] = df['tweet'].apply(lambda x: list(
        nltk.everygrams(x.split(' '), min_len=min_len, max_len=max_len))
    )
    return df


def get_topic_words(
        df: pd.DataFrame,
        col: str,
        vectoriser,
        decomposer,
):
    term_matrix = vectoriser.fit_transform(df[col].values.astype('U'))

    decomposer.fit(term_matrix)
    return vectoriser, decomposer, term_matrix


def print_topic_words(decomposer, vectoriser, n_words):
    for i, topic in enumerate(decomposer.components_):
        print(f'Top {n_words} words for topic #{i}:')
        print([vectoriser.get_feature_names_out()[i]
               for i in topic.argsort()[-n_words:]])
        print('\n')


def find_optimal_clusters(data, max_k):
    """
    Iterate through each cluster size up to max_k and plot SSE of each clusters
    """
    k_vals = range(2, max_k + 1, 2)
    
    sse = []
    for k in k_vals:
        sse.append(
            KMeans(n_clusters=k, random_state=520).fit(data).inertia_
            
            #MiniBatchKMeans(
            #    n_clusters=k, init_size=1024, batch_size=2040, random_state=20
            #).fit(data).inertia_
        )
        print(f"Fitted {k} clusters!")
    
    
    # Plot the graph
    f, ax = plt.subplots(1, 1)
    ax.plot(k_vals, sse, marker='o')
    ax.set_xlabel('Cluster Centres')
    ax.set_xticks(k_vals)
    ax.set_xticklabels(k_vals)
    ax.set_ylabel('SSE')


def plot_clusters(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=3000)

    pca = PCA(n_components=2).fit_transform(
        np.asarray(data[max_items, :].todense())
    )

    tsne = TSNE().fit_transform(
        PCA(n_components=50).fit_transform(
            np.asarray(data[max_items, :].todense())
        )
    )
    
    idx = np.random.choice(range(pca.shape[0]), size=2000, replace=False)

    label_subset = labels[max_items]
    label_subset = [cm.hsv(i/max_label) for i in label_subset[idx]]
    
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title("PCA Clusters")
    ax[1].scatter(pca[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title("TSNE Clusters")


"""
TODO: Bag-of-words 
LIWC

TUI
jupyter
OOP - refactor
send kaggle email


look at keywords and raed tweet - uncommon words
- freq of tweets containing keyword
- sentiment as dimension in clustering
- conduct on subset of tweet
- hedonometre - compare with VADER
    - gorup of tweets
- labMT package
- 
"""




