import string

from nlp import *

from nltk import TweetTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation


data_file = "../Dataset/twitter.csv"

init_nltk_downloads()
df = load_twitter_csv(data_file)
stop_words = stopwords.words('english')

stop_words.extend(
    list(string.punctuation) + ['would', 'could', 'get',
                                'want', 'he', 'twitter']
)

tokeniser = TweetTokenizer()
lemmatiser = WordNetLemmatizer()

df['tweet'] = df['tweet'].apply(
    furnish, args=(tokeniser, lemmatiser, stop_words)
)


tfidf_vec, nmf, tfidf_term_matrix = \
    get_topic_words(df, 'tweet', TfidfVectorizer, NMF)
count_vec, lda, count_term_matrix = \
    get_topic_words(df, 'tweet', CountVectorizer, LatentDirichletAllocation)

print_topic_words(nmf, tfidf_vec, 15)
print_topic_words(lda, count_vec, 15)


print(":D")
