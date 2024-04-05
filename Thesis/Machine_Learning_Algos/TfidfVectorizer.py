from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()

tfidf = vec.fit_transform(['I like machine learning and clustering algorithms',
                           'Apples, oranges and any kind of fruits are healthy',
                           'Is it feasible with machine learning algorithms?',
                           'My family is happy because of the healthy fruits'])

# similarity matrix
print((tfidf*tfidf.T).A)