from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=1)

#print("\n".join(training_data.data[0].split("\n")[:30]))
#print("Target is:", training_data.target_names[training_data.target[0]])

# we just count the word occurrences
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)
#print(count_vector.vocabulary_)

# we transform the word occurrences into tf-idf
# TfidfVectorizer = CountVectorizer + TfidfTransformer
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

model = MultinomialNB().fit(x_train_tfidf, training_data.target)


# TESTING MODEL
new = ['This has nothing tod with church or religion', 'Software engineering is getting hotter and hotter', 'My favorite topic has something to do with Quantum Physics']

x_new_counts = count_vector.transform(new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

predicted = model.predict(x_new_tfidf) 

for doc, category in zip(new, predicted):
    print("%r ------> %s" % (doc, training_data.target_names[category]))

