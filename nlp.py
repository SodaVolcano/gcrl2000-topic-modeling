"""
Contains NLP functions with a focus on topic modelling of tweets

Most of the functions focus on loading, preprocessing, and visualising the
textual data. Example of how the functions are used can be found in the
``main()`` function which is run when this file is run as a script.

- Author: Tin Chi Pang (23301921)
"""

# Native imports
from typing import Literal, Any
import re       # Regex matching
import string   # For string.punctuations

# 3rd party imports
import nltk  # Natural language toolkit
from nltk import TweetTokenizer, WhitespaceTokenizer, WordNetLemmatizer, \
    pos_tag
from nltk.corpus import stopwords

import scipy.sparse   # For issparse() - check if matrix is sparse
import pandas as pd   # For csv data storage and manipulation
import numpy as np

# Contains vectoriser and decomposer models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, LatentDirichletAllocation
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
) -> str:
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
        txt_col: str,
        stop_words: list[str] = stopwords.words('english'),
        tokeniser: Any = TweetTokenizer(),
        lemmatiser: Any = WordNetLemmatizer(),
        filter_regex: str = r'',
        remove_punct: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = True,
        remove_urls: bool = True,
        remove_duplicates: bool = True,
        casing: Literal["lower", "upper"] | None = 'lower',
        inplace: bool = False
) -> pd.DataFrame:
    """
    Preprocess specified column of input DataFrame.

    The preprocessed texts is stored in a new column that's attached to the
    input DataFrame. The column is named by appending ``"_preprocessed"`` onto
    the end of the original column's name.

    Parameters
    ----------
    df : ``pandas.DataFrame`` that contains the textual data.
    txt_col : Name of the column in DataFrame containing the textual data.
    stop_words : List of words to filter out from the textual data. By default,
        the English stop word set from ``nltk.corpus.stopwords`` is used
        (requires the nltk resource ``stopwords`` to be downloaded).
    tokeniser : Object that implements ``tokenize(text: str)``; breaks up
        texts into lists of words. By default, ``nltk.TweetTokenizer`` is used.
    lemmatiser : Object that implements ``lemmatize(word: str, pos_tag: str)``;
        reduces words to their dictionary form. For example, the dictionary
        form of ['break', 'broken', 'breaks', 'breaking'] is 'break'. By
        default, ``nltk.WordNetLemmatizer`` is used (requires the nltk
        resource ``wordnet`` to be downloaded). The nltk resource
        ``averaged_perceptron_tagger`` is required regardless whether the
        default lemmatiser is used or not.
    filter_regex : Regex string that match patterns in the text to be removed.
    remove_punct : Set to ``True`` to remove punctuations from the text.
    remove_mentions : Set to ``True`` to remove Twitter mentions (defined as
        any alphanumeric words that may be delimited by "_" and begins with
        "@").
    remove_hashtags : Set to ``True`` to remove hashtags (defined as any
        alphanumeric words that may be delimited by "_" and begins with "#").
    remove_urls : Set to ``True`` to remove URLs.
    remove_duplicates : Set to ``True`` to drop any duplicate texts in the
    **raw dataset** (duplicate texts may be present after preprocessing).
    casing : Convert all letters to lowercase or uppercase; set to ``None``
        to preserve current casing.
    inplace : Set to ``True`` to alter the input DataFrame without produce a
        new one. By default, ``inplace=False`` and a new DataFrame is produced.

    Returns
    -------
    DataFrame
        A two-dimensional tabular data with the preprocessed texts stored in a
        new column named ``"<txt_col>_preprocessed"``.
    """

    if not inplace:
        df = df.copy()

    filter_regex = _build_regex(
        filter_regex, remove_mentions, remove_hashtags, remove_urls
    )

    if remove_duplicates:
        df[txt_col].drop_duplicates(inplace=True)

    casing_func = None
    if casing == 'lower':
        casing_func = str.lower
    elif casing == 'upper':
        casing_func = str.upper
    elif casing is not None:
        raise ValueError(
            "Parameter 'casing' can only have value: 'lower', 'upper', or None"
        )

    new_col = txt_col + "_preprocessed"

    # Apply regex filter - remove all matched texts
    if len(filter_regex) > 0:
        if casing_func is None:
            df[new_col] = df[txt_col].apply(
                lambda txt: re.sub(filter_regex, '', txt)
            )
        else:
            df[new_col] = df[txt_col].apply(
                lambda txt: casing_func(re.sub(filter_regex, '', txt))
            )

    df[new_col] = df[new_col].apply(
        furnish, args=(remove_punct, tokeniser, lemmatiser, stop_words)
    )

    return df


def furnish(
        text: str,
        remove_punct: bool,
        tokeniser: Any = WhitespaceTokenizer(),
        lemmatiser: Any = WordNetLemmatizer(),
        stop_words: list[str] = stopwords.words('english')
) -> str:
    """
    Lemmatise words and remove stopwords from a single string.

    Lemmatisation involves reducing a word to its dictionary form. For
    example, the dictionary form of ['break', 'broken', 'breaks',
    'breaking'] is 'break'.

    Requires the nltk resource ``averaged_perceptron_tagger``.

    Parameters
    ----------
    text : String to be preprocessed.
    remove_punct : Set to ``True`` to remove punctuations from the text,
        defined as any non-alphanumeric characters.
    tokeniser : Object that implements ``tokenize(text: str)``; breaks up
        texts into lists of words. By default, ``nltk.TweetTokenizer`` is used.
    lemmatiser : Object that implements ``lemmatize(word: str, pos_tag: str)``;
        reduces words to their dictionary form. By default,
        ``nltk.WordNetLemmatizer`` is used (requires the nltk
        resource ``wordnet`` to be downloaded).
    stop_words : List of words to filter out from the textual data. By default,
        the English stop word set from ``nltk.corpus.stopwords`` is used
        (requires the nltk resource ``stopwords`` to be downloaded).

    Returns
    -------
    String with words lemmatised and stop words removed.
    """
    final_text = []
    for word, tag in pos_tag(tokeniser.tokenize(text)):
        # Tag word as verb, nouns, etc., improves lemmatiser accuracy
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
        drop_linked_tweets: bool = False,
        **pd_read_csv_args
) -> pd.DataFrame:
    """
    Load CSV file containing tweets into a DataFrame and preprocess the data.

    Input CSV file must have the column "tweet". Use the normal
    ``pandas.DataFrame()`` constructor for other CSV files. The tweets will
    be preprocessed using preset settings if ``do_preprocess=True``. Set it
    to ``False`` if you want more control over the settings.

    Parameters
    ----------
    file_path : String path of the CSV file to load from.
    usecols : List of column names to load into the resulting DataFrame. If
        ``None``, the default columns ``['conversation_id', 'tweet',
        'language']`` will be used.
    index_col : Column number in the CSV file to use as the index column in
        the DataFrame.
    eng_only : Drop all non-English tweets. if ``True``, the file MUST have a
        column named "language" with value "en" indicating an English tweet.
    do_preprocess : Set to ``True`` to preprocess tweets with default params.
    drop_linked_tweets : Set to ``True`` to drop any tweets that have a URL -
        this is intended to remove "news bots" that spam the same headline
        with a link to a news article, which can influence term frequency. This
        argument filters the tweets to only those made by users not
        promoting anything using URLs.
    pd_read_csv_args : Additional arguments passed to ``pandas.read_csv()``.
        One useful argument is ``nrows=n`` which reads the first ``n`` rows
        in the CSV file.

    Returns
    -------
    DataFrame with content of the CSV file.
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

    if drop_linked_tweets:
        regex_urls = \
            r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\." + \
            r"[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()!@:%_\+.~#?&\/\/=]*)"

        # pandas throws a warning when using str.contains(regex)
        import warnings
        warnings.filterwarnings(
            "ignore", "This pattern is interpreted as a regular expression"
        )

        # get Series of bool on if each row matched with regex
        regex_matches = df['tweet'].str.contains(regex_urls)
        # ~ = invert operation, we delete all rows that DID match
        df = df[~regex_matches]

    if do_preprocess:
        preprocess_df(df, txt_col='tweet', inplace=True)
    return df


# ========================= Visualisation =============================

def plot_document_matrix(
        doc_matrix: Any,
        decomposer: Literal["pca", "tsne"] | Any = "pca",
        dimension: Literal[2, 3] = 2,
        n_samples: int = 3000,
        **decomposer_args
) -> None:
    """
    Plot a matrix in 2D or 3D space.

    If a document-terms matrix is inputted, the plotted points will be each
    documents in the vector space where documents with similar terms will be
    closer in distance.

    If a document-topics matrix produced by LDA is inputted, the plotted
    points will be each document in a Dirichlet distribution with the
    corners being the topics. Note that the number of topics will be reduced to
    3 or 4 if dimension is 2D or 3D respectively.

    **Important** : If this function is called as a .py script (instead of
        being in a jupyter notebook), you must call
        ``matplotlib.pyplot.show()`` to display the graph.

    Parameters
    ----------
    doc_matrix : Matrix to plot; a document-terms matrix or document-topics
        matrix.
    decomposer : String that indicate which dimensionality reduction
        technique to use - ``"pca"`` for Principal Component Analysis, or
        ``tsne`` for $t$-distributed Stochastic Neighbour Embedding. To
        visualise high-dimensional data in 2D or 3D plane, it's necessary to
        reduce the dimension. You can pass in your own decomposer as the
        parameter instead of a string. By default, ``"pca"`` is used.

        **NOTE**: ``sklearn.decomposition.TSNE`` has a hyperparameter
        ``perplexity`` that is often manipulated to change how data points
        are unfolded. This can be passed to this function like normal
        keyword arguments.
    dimension : Number of axis to visualise the data on, 2D or 3D. By default,
        data will be plotted on a 2D plane.
    n_samples : Number of sample points to plot, 3000 by default.
    decomposer_args : Any additional keyword parameters not specified in the
        function signature will be passed to the decomposer.
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
        transformed = PCA(
            n_components=dimension, **decomposer_args
        ).fit_transform(doc_matrix)

    elif decomposer == 'tsne':
        # Reduce dimension to 50 before using t-SNE
        if doc_matrix.shape[1] > 50:
            doc_matrix = PCA(n_components=50).fit_transform(doc_matrix)
        transformed = TSNE(
            n_components=dimension, init='pca',
            early_exaggeration=50, **decomposer_args
        ).fit_transform(doc_matrix)
    else:
        transformed = decomposer.fit_transform(doc_matrix)

    fig = plt.figure()
    if dimension == 2:
        ax = fig.add_subplot(projection=None)
        ax.scatter(transformed[:, 0], transformed[:, 1])
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


def print_topic_terms(
        topic_terms: Any,
        vocab: list[str],
        n_words: int = 10,
        topic_idx: int | None = None,
        dump_to_file: bool = False,
        dump_file_name: str = "./topic_terms_dump.txt"
) -> None:
    """
    Print the term distribution for each topic in the topic-terms matrix.

    Parameters
    ----------
    topic_terms: Topic-terms matrix showing term distribution for each topic.
    vocab : List of terms in the matrix.
    n_words : Number of words per topic to print.
    topic_idx : Index of the topic to print the terms for, set to ``None`` to
        print terms for all topics (default).
    dump_to_file : Set to ``True`` to write all output to a text file instead
        of printing to console. ``False`` by default.
    dump_file_name : Name of the dump file to write to when
        ``dump_to_file=True``. By default, all output is printed to
        ``"./topic_terms_dump.txt"``.
    """
    file = open(dump_file_name, "w") if dump_to_file else None

    # Only print ONE topic; put into a list to bypass enumerate()
    if topic_idx is not None:
        topic_terms = [topic_terms[topic_idx]]

    for i, topic in enumerate(topic_terms):
        if topic_idx is not None:
            i = topic_idx
        print(f'Top {n_words} words for topic #{i}:', file=file)
        print([vocab[i] for i in topic.argsort()[-n_words:]][::-1], file=file)
        print('\n', file=file)

    if file is not None:
        file.close()


def print_doc_topics(
        doc_topics: Any,
        corpus: Any,
        n_docs: int = 30,
        topic_idx: int | None = None,
        dump_to_file: bool = False,
        dump_file_name: str = "./doc_topics_dump.txt"
) -> None:
    """
    For each topic, print the documents with highest % mixture for that topic.

    LDA assumes each document consists of a mixture of topics (e.g. 40%
    topic 1, 33% topic 2, ...). For each topic, this function finds the
    documents with the highest mixture score for that topic and prints them.

    Parameters
    ----------
    doc_topics : Document-topics matrix showing topic mixture for each
        document.
    corpus : Collection of documents used to create the document-topics
        matrix. Ideally, this should be the unprocessed corpus where the texts
        are more readable.
    n_docs : Number of documents to print per topic, 30 by default.
    topic_idx : Index of the topic to print the documents for,
        set to ``None`` to print for all topics (default).
    dump_to_file : Set to ``True`` to write all output to a text file instead
        of printing to console. ``False`` by default.
    dump_file_name : Name of the dump file to write to when
        ``dump_to_file=True``. By default, all output is printed to
        ``"./doc_topics_dump.txt"``.
    """
    corpus = list(corpus)
    # Swap the axis, produce topic-documents matrix
    topic_docs = [doc_topics[:, i] for i in range(doc_topics.shape[1])]

    file = open(dump_file_name, "w") if dump_to_file else None

    # Only print for ONE topic; put into a list to bypass enumerate()
    if topic_idx is not None:
        topic_docs = [topic_docs[topic_idx]]

    for i, topic in enumerate(topic_docs):
        if topic_idx is not None:
            i = topic_idx
        print('=' * 80 + '\n', file=file)
        print(f'Top {n_docs} documents for topic #{i}:\n\n', file=file)

        doc_idx = topic.argsort()[-n_docs:][::-1]
        for doc in doc_idx:
            print(f'Topic %: {topic[doc]:.3f}\nDocument: '
                  f'{corpus[doc]}\n', file=file)
        print(file=file)

    if file is not None:
        file.close()


def graph_topic_terms_matrix(
        topic_terms: Any,
        vocab: Any,
        topic_idx: int,
        max_words: int = 30,
) -> None:
    """
    Graph the term distribution for a given topic.

    Parameters
    ----------
    topic_terms : Matrix of shape ``(n_topics, n_terms)`` where each topic
        is a vector of ``n_terms``-dimension, with the $i$-th value indicating
        the weight of the $i$-th term in the vocabulary towards that topic.
    vocab : A list of all unique terms.
    topic_idx : Index of the topic in ``topic_terms`` to graph for.
    max_words : Number of words to graph, clipped to the size of the
        vocabulary.
    """
    topic = topic_terms[topic_idx]

    if max_words > len(vocab):
        max_words = len(vocab)

    # list of index of top most frequent terms
    top_terms = [i for i in np.argsort(topic)[-max_words:]]

    frequencies = [topic[i] for i in top_terms]

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar([vocab[i] for i in top_terms][::-1], frequencies[::-1])
    ax.set_ylabel("Term Weighting")
    fig.autofmt_xdate(rotation=90, ha='right')  # Tilt the x labels
    ax.set_title(f"Term Probability Distribution For Topic {topic_idx}")


# ========================================================================

def main():
    """
    Demonstrates the usage of various ``nlp.py`` functions.
    """
    resources = ['corpora/stopwords', 'corpora/wordnet',
                 'taggers/averaged_perceptron_tagger']

    print("Checking installation of nltk resources...")
    check_nltk_resources(resources)

    default_stopwords = stopwords.words('english')
    default_stopwords.extend(
        list(string.punctuation) + [
            'would', 'could', 'get', 'want', 'he', 'twitter', 'elon', 'musk',
            'well', 'need', 'come', 'really', 'take', 'say', 'go', 'use',
            'make', 'know', 'think'
        ]
    )

    file = "./dataset/twitter.csv"
    print(f"Loading dataset from {file}...")
    df = load_twitter_csv(file, do_preprocess=False, drop_linked_tweets=True)

    print("Preprocessing the textual data...")
    preprocess_df(df, txt_col='tweet', stop_words=default_stopwords,
                  inplace=True)

    corpus = df['tweet_preprocessed']

    print("Vectorising the textual data using bag-of-words with min_df=30, "
          "max_df=0.95...")
    vect = CountVectorizer(min_df=30, max_df=0.95)
    doc_term = vect.fit_transform(corpus)

    print("Training LDA model on the vectorised data with n_components=7, "
          "max_iter=10...")
    model = LatentDirichletAllocation(n_components=7, max_iter=10)
    model_matrix = model.fit_transform(doc_term)

    print_topic_terms(model.components_, vect.get_feature_names_out(),
                      n_words=10, dump_to_file=False)

    print("Dumping top most relevant documents per topic to text file...")
    print_doc_topics(model_matrix, df['tweet'], n_docs=50, dump_to_file=True)

    print("Plotting resulting document matrix using PCA...")
    plot_document_matrix(model_matrix, decomposer='pca')
    plt.show()


if __name__ == "__main__":
    main()
