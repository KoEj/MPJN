import csv
import re

from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

labels = []
content = []

originalTokens = []
withoutStopwordsTokens = []
stemmerTokens = []
lemmatizeTokens = []

tokensGrouped = {
    "originalTokens": originalTokens,
    "withoutStopwordsTokens": withoutStopwordsTokens,
    "stemmerTokens": stemmerTokens,
    "lemmatizeTokens": lemmatizeTokens
}


# Zadanie 2.1
with open("imbd.csv", encoding="utf8", newline='') as my_file:
    reader = csv.DictReader(my_file)

    for row in reader:
        # print(row['label'], row['content'])
        content.append(row['content'].lower())
        labels.append(row['label'])

t = TreebankWordTokenizer()
ps = PorterStemmer()
wnl = WordNetLemmatizer()
stopwords = stopwords.words('english')

for contentLine in content:
    originalToken = t.tokenize(contentLine)
    originalTokens.append(originalToken)
    # print(originalToken)

    newContentLine = re.sub(r'[\,\-\—\.\(\)\'\`\.\;]', '', contentLine)
    withoutStopwordsToken = t.tokenize(newContentLine)
    withoutStopwordsToken = [w for w in withoutStopwordsToken if w.lower() not in stopwords]
    # print(withoutStopwordsToken)
    withoutStopwordsTokens.append(withoutStopwordsToken)

    wordsArray = []
    for word in withoutStopwordsToken:
        wordsArray.append(ps.stem(word))

    stemmerTokens.append(wordsArray)
    # print(wordsArray)

    wordsArray = []
    for word in withoutStopwordsToken:
        wordsArray.append(wnl.lemmatize(word))

    lemmatizeTokens.append(wordsArray)
    # print(wordsArray)


# Zadanie 2.2
def toStringAndVectorize(tokens):
    newArray = []
    for row in tokens:
        s = ' '.join([x for x in row])
        newArray.append(s)

    return newArray


vectorizer = CountVectorizer()
for group in tokensGrouped:
    X_train, X_test, y_train, y_test = train_test_split(tokensGrouped[group], labels, test_size=0.30, random_state=1337)

    X_train = vectorizer.fit_transform(toStringAndVectorize(X_train))
    X_test = vectorizer.transform(toStringAndVectorize(X_test))

    clf = MultinomialNB(force_alpha=True)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print(group, accuracy_score(y_test, pred))

# originalTokens 0.16111111111111112
# withoutStopwordsTokens 0.16111111111111112
# stemmerTokens 0.16111111111111112
# lemmatizeTokens 0.16111111111111112
# Wyniki wyszły dosyć słabe, nieróżniące się od siebie. Operacje przeporwadzone powinny w teorii polepszać jakość klasyfikacji, jednak tak nie jest.

# Zadanie 2.3
def prepareX(f_group):
    newArray = []
    for row in f_group:
        # print(row)
        s = ' '.join(row)
        newArray.append(s)
    return newArray

print('--------------------------------------------------')
tfidVect = TfidfVectorizer(max_features=300)
labels = np.array(labels).astype(int)
for group in tokensGrouped:
    scores = []
    X = tfidVect.fit_transform(prepareX(tokensGrouped[group]))
    skf = StratifiedKFold(n_splits=5)

    for train_index, test_index in skf.split(X, labels):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        mlpcClass = MLPClassifier()
        mlpcClass.fit(x_train, y_train)
        pred_y = mlpcClass.predict(x_test)

        scores.append(accuracy_score(y_test, pred_y))
        print(accuracy_score(y_test, pred_y))
    print(group, scores)



