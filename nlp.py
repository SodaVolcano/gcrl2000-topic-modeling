"""
Uses Non-Negative matrix Factorisation (NMF)
"""
import re

import pandas as pd
import nltk
from nltk import pos_tag


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
def furnish(text: str, tokeniser, lemmatiser, stop_words: list):
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


def get_topic_words(
        df: pd.DataFrame, col: str, vectoriser_init, decomposer_init
):
    vectoriser = vectoriser_init(max_df=0.95, min_df=4)
    term_matrix = vectoriser.fit_transform(df[col].values.astype('U'))

    decomposer = decomposer_init(n_components=4)
    decomposer.fit(term_matrix)
    return vectoriser, decomposer, term_matrix


def print_topic_words(decomposer, vectoriser, n_words):
    for i, topic in enumerate(decomposer.components_):
        print(f'Top {n_words} words for topic #{i}:')
        print([vectoriser.get_feature_names_out()[i]
               for i in topic.argsort()[-n_words:]])
        print('\n')


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




