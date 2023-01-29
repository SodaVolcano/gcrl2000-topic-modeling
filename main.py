from nlp import init_nltk_downloads, load_twitter_csv, furnish, get_topic_words, print_topic_words

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation



data_file = "../Dataset/twitter.csv"

init_nltk_downloads()
df = load_twitter_csv(data_file)


df['tweet'] = df['tweet'].apply(furnish)

#df = gen_n_grams(df)

tfidf_vec, nmf, tfidf_term_matrix = get_topic_words(
    df, 'tweet',
    TfidfVectorizer(max_df=0.95, min_df=4),
    NMF(n_components=4)
)

count_vec, lda, count_term_matrix = get_topic_words(
    df, 'tweet', CountVectorizer(max_df=0.95, min_df=4),
    LatentDirichletAllocation(n_components=4)
)

print_topic_words(nmf, tfidf_vec, 15)
print_topic_words(lda, count_vec, 15)


print(":D")