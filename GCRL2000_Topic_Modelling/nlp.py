"""
Contains NLP functions with a focus on topic modelling of tweets
"""

# Native imports
from typing import Iterable, Literal, Any
import re       # Regex matching
import string   # For string.punctuations

# 3rd party imports
import nltk  # Natural language toolkit
import scipy.sparse
from nltk import TweetTokenizer, WhitespaceTokenizer, WordNetLemmatizer, \
    pos_tag
from nltk.corpus import stopwords

import pandas as pd   # For csv data storage and manipulation
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,\
    HashingVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA, \
    NMF, TruncatedSVD
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt  # For visualisation


# ====================== Misc =========================================

def check_nltk_resources(resources: list[str]) -> None:
    """
    Given a list of resources, find and download them for nltk if needed
    """
    for resource in resources:
        # Find .zip file instead since nltk have problem unzipping files
        try:
            nltk.find(f'{resource}.zip')
        except LookupError:
            nltk.download(resource.split('/')[-1])


def _build_regex(
        regex: str, remove_mentions: bool,
        remove_hashtags: bool, remove_urls: bool
):
    """
    Append regex strings to input regex according to parameters.

    Three regex strings are available, each matching Twitter mentions,
    hashtags, and URLs. Each regex attached is delimited by "|".
    """
    regex_mentions = r"(@[A-Za-z0-9_]+)"
    regex_hashtags = r"(#[A-Za-z0-9_]+)"
    regex_urls = \
        r"(https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\." + \
        r"[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*))"

    # Keep as list to reduce redundant code
    regexes = [regex_mentions, regex_hashtags, regex_urls]
    add_regex = [remove_mentions, remove_hashtags, remove_urls]

    for i in range(len(regexes)):
        # Add '|" onto the end if regex string is not empty
        regex = (lambda r: f"{r}{r'|' if len(r) != 0 else r''}")(regex)
        if add_regex[i]:
            regex += regexes[i]

    return regex


# ====================== Data Preprocessing ============================

def preprocess_df(
        df: pd.DataFrame,
        txt_col: str = "tweet",
        stop_words: list[str] = stopwords.words('english'),
        tokeniser=TweetTokenizer(),
        lemmatiser=WordNetLemmatizer(),
        filter_regex: str = r'',
        remove_punct: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = True,
        remove_urls: bool = True,
        casing: Literal["lower", "upper", None] = 'lower',
        inplace=False
) -> pd.DataFrame:
    """
    Conduct text preprocessing on a DataFrame's column, e.g. remove stop words

    :param df: DataFrame containing the textual data
    :param txt_col: Column of the DataFrame to process, "tweet" by default
    :param stop_words: List of words to remove from the documents
    :param tokeniser: Object implementing `tokenize(text: str)` that
        breaks up `text` into a list of words, `TweetTokeniser` by default
    :param lemmatiser: Object implementing `lemmatize(word, pos_tag)`
        that reduces `word` into its root form (e.g. "asking" -> "ask").
        `pos_tag` is a string denoting what part-of-speech a word is, e.g. 'v'
        for verbs, 'n' for nouns, this improves lemmatiser's accuracy.
        `WordNetlemmatizer` by default.
    :param filter_regex: Regex string that match patterns in texts to remove.
    :param remove_punct: Set `True` to remove punctuations from text. This adds
        `string.punctuation` to the list of stop words.
    :param remove_mentions: Set `True` to remove Twitter mentions (any
        alphanumeric words beginning with "@" and may contain "_").
    :param remove_hashtags: Set `True` to remove Twitter mentions (any
        alphanumeric words beginning with "#" and may contain "_").
    :param remove_urls: Set `True` to remove URLs from text
    :param casing: Specify lowercasing or uppercasing, set to `None` to disable

    :return: DataFrame containing the preprocessed texts
    """
    if not inplace:
        df = df.copy()
    
    filter_regex = _build_regex(
        filter_regex, remove_mentions, remove_hashtags, remove_urls
    )

    # Empty function, don't change casing
    def casing_func(x):
        return x

    if casing == 'lower':
        casing_func = str.lower
    elif casing == 'upper':
        casing_func = str.upper
    elif casing is not None:
        raise ValueError(
            "Parameter 'casing' can only have value: 'lower', 'upper', or None"
        )

    if len(filter_regex) > 0:
        # Apply regex filter - remove all matched texts
        df[txt_col] = df[txt_col].apply(
            lambda txt: casing_func(re.sub(filter_regex, '', txt))
        )

    df[txt_col] = df[txt_col].apply(
        furnish, args=(remove_punct, tokeniser, lemmatiser, stop_words)
    )

    return df


def furnish(
        text: str,
        remove_punct: bool,
        tokeniser=WhitespaceTokenizer(),
        lemmatiser=WordNetLemmatizer(),
        stop_words: list[str] = stopwords.words('english')
) -> str:
    """
    Preprocess each text string, tokenise, lemmatise, and remove stop words

    Input `text` is split into tokens and each token is reduced to their
    root form using the input tokeniser and lemmatiser respectively. Before
    lemmatisation, each token is tagged to a part-of-speech category such as
    nouns or verbs - this improves the lemmatiser's accuracy.

    :return: Preprocessed text created by joining remaining tokens by space.
    """
    final_text = []
    for word, tag in pos_tag(tokeniser.tokenize(text)):
        # Tag word as verb, nouns, etc, improves lemmatiser accuracy
        tag = tag.lower()[0]
        if tag in ['a', 'r', 'n', 'v']:
            word = lemmatiser.lemmatize(word, tag)
        else:
            word = lemmatiser.lemmatize(word)

        if word not in stop_words and (remove_punct and word.isalnum()):
            final_text.append(word)

    return ' '.join(final_text)


# ====================== Data Preprocessing ============================

def load_twitter_csv(
        file_path: str,
        usecols: list[str] = None,
        index_col: int = 0,
        eng_only: bool = True,
        do_preprocess: bool = True,
        **pd_read_csv_args
) -> pd.DataFrame:
    """
    Preset function to load and preprocess tweets from CSV file into DataFrame

    Input CSV file must have the column "tweet". If `eng_only` is specified,
    the column "language" is required with value "en" indicating that a tweet
    is in English. Returned DataFrame will have the tweets preprocessed if
    `do_preprocess` is set to `True` (e.g. removing stop words etc.).

    For more customisation, manually create a DataFrame and call
    `preprocess()`.

    :param file_path: Path of the CSV file to read form
    :param usecols: List of column names to load. By default, columns
        "conversation_id", "tweet", and "language" are used.
    :param index_col: Index of the column to use index for the DataFrame,
        0 by default.
    :param eng_only: Set to `True` to discard all non-English tweets. Require
        the column "language" with value "en" denoting English Tweets. The
        column "language" is discarded from the final DataFrame.
    :param do_preprocess: Set to `True` to preprocess the tweets.
    """
    if usecols is None:
        usecols = ['conversation_id', 'tweet', 'language']

    df = pd.read_csv(
        file_path, index_col=index_col, usecols=usecols, **pd_read_csv_args
    )

    # Filter out non-English tweets
    if eng_only:
        df.query('language == "en"', inplace=True)
        df.drop(columns=['language'], inplace=True)

    df.dropna(subset=['tweet'], inplace=True)

    if do_preprocess:
        df = preprocess_df(df)
    return df


# ========================= Visualisation =============================

def plot_document_matrix(
        doc_matrix,
        decomposer: Literal["pca", "tsne"] | Any = "pca",
        perplexity: float = 30.0,
        dimension: Literal[2, 3] = 2,
        n_samples: int = 3000
) -> None:
    """
    Plot the documents in a term matrix in 2D or 3D space.

    Documents containing similar terms will be closer on the graph.
    
    **IMPORTANT**: Call `matplotlib.pyplot.show()` to display the graph.

    :param matrix: Matrix to plot; can be document-term or document-topic
        matrix
    :param decomposer: Str indicating which dimensionality reduction technique
        to use. Pass in non-string to specify own decomposer. "pca" by default
    :param perplexity: Hyperparameter used for t-SNE, ignored if using PCA
    :param dimension: Number of dimension in the resulting plot, i.e. 2D or 3D
    :param n_samples: Number of points from term matrix to plot
    """
    if dimension not in [2, 3]:
        raise ValueError(
            "Parameter 'dimension' can only have value 2 or 3"
        )

    if n_samples > doc_matrix.shape[0]:
        n_samples = doc_matrix.shape[0]

    # Randomly sample some documents to plot
    samples = np.random.choice(
        range(doc_matrix.shape[0]), size=n_samples, replace=False
    )

    # Convert data to the appropriate type and sample size
    if scipy.sparse.issparse(doc_matrix):
        doc_matrix = doc_matrix.todense()
    
    doc_matrix = np.asarray(doc_matrix[samples, :])

    if decomposer == 'pca':
        doc_matrix = np.asarray(doc_matrix)
        transformed = \
            PCA(n_components=dimension).fit_transform(doc_matrix)
    elif decomposer == 'tsne':
        if doc_matrix.shape[1] > 50:
            doc_matrix = PCA(n_components=50).fit_transform(doc_matrix)
        transformed = TSNE(
            n_components=dimension, init='pca', verbose=1,
            early_exaggeration=50, perplexity=perplexity
        ).fit_transform(doc_matrix)
    else:
        transformed = decomposer.fit_transform(doc_matrix)

    # TODO: Add PickEvent
    fig = plt.figure()
    if dimension == 2:
        ax = fig.add_subplot(projection=None)
        line_2d = ax.scatter(
            transformed[:, 0], transformed[:, 1]#, picker=True, pickradius=5
        )
    else:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2])
        ax.set_zlabel('Component 3')

    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_title(
        f"Documents in {dimension}D Embedded Space "
        f"({decomposer.upper()})"
        "\n(Similar documents are closer in space)"
    )


def print_topics(model, vectoriser, n_words):
    terms = vectoriser.get_feature_names_out()
    
    for i, topic in enumerate(model.components_):
        print(f'Top {n_words} words for topic #{i}:')
        print([terms[i] for i in topic.argsort()[-n_words:]][::-1])
        print('\n')


def graph_topic_terms_matrix(
        topic_terms, vocab, topic_idx: int, max_words: int = 30,
):
    """
    Graph the term probability distribution for a given topic

    :param topic_terms: Matrix of shape (n_topics, n_terms) where each topic
        is a vector of n_terms values, with the i-th value indicating weight of
        the i-th term in vocabulary towards that topic.
    :param vocab: list of all terms
    :param topic_idx: Index of the topic in topic_terms to graph
    :param max_words: Number of words to graph, clipped to size of vocab
    """
    topic = topic_terms[topic_idx]

    if max_words > len(vocab):
        max_words = len(vocab)

    # list of index of top most frequent terms
    top_terms = [i for i in np.argsort(topic)[-max_words:]]

    frequencies = [topic[i] for i in top_terms]
    #total_freq = sum(frequencies)
    #percentages = [(i / total_freq) for i in frequencies]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar([vocab[i] for i in top_terms][::-1], frequencies[::-1])
    ax.set_ylabel("Term Weighting")
    fig.autofmt_xdate(rotation=90, ha='right')  # Tilt the x labels
    ax.set_title(f"Term Probability Distribution For Topic {topic_idx}")


# ======================== Testing Ground =============================

if __name__ == "__main__":
    resources = ['corpora/stopwords', 'corpora/wordnet',
                'taggers/averaged_perceptron_tagger']

    check_nltk_resources(resources)

    default_stopwords = stopwords.words('english')

    default_stopwords.extend(
        list(string.punctuation) + [
            'would', 'could', 'get', 'want', 'he', 'twitter', 'elon', 'musk',
            'well', 'need', 'come', 'really', 'take', 'say', 'go', 'use', 
            'make', 'know', 'think'

        ]
    )

    df = load_twitter_csv("./dataset/twitter.csv", do_preprocess=False)
    preprocess_df(df, stop_words=default_stopwords, inplace=True)

    corpus = df['tweet']
    #corpus = ['dog and cat are antagonistic', 'dog and dog dislike chocolate',
    #        'chocolate is bad for cat as well', 'dog dislike cat',
    #        'cat dislike dog', 'dog dislike cat too']

    vect = CountVectorizer(min_df=30, max_df=0.95)
    doc_term = vect.fit_transform(corpus)

    model = NMF(n_components=7, max_iter=20)
    model_matrix = model.fit_transform(doc_term)
    
    print_topics(model, vect, 15)

    plot_document_matrix(model_matrix, decomposer='pca')
    plt.show()


"""
VERY IMPORTANT
    term_matrix is of shape (n_documents, n_terms)
PCA is reducing matrix by n_terms, NOT n_documents
    So n_documents stays the SAME, but n_terms REDUCE

Plot visualisation:
    Dots = document
    axis (PC) = words weight
    
"""