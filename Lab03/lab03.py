import os
import numpy as np
from nltk import word_tokenize
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Zadanie 3.1
os.chdir('poezja/')
dir_names = ['adam', 'jan', 'juliusz']

file_names = []
for dir_name in dir_names:
    for file_name in os.listdir(dir_name):
        if file_name.endswith('.txt'):
            file_names.append(os.path.join(dir_name, file_name))

with open('stopwordsPL.txt', 'r', encoding='utf-8') as f:
    stopwords = f.read().split()

stop_words = set(stopwords)

documents = {'adam': [], 'jan': [], 'juliusz': []}
document_list = []
filtered_tokens = {}

for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
        tokens = word_tokenize(text.lower())
        f_tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
        filtered_tokens[file_name] = f_tokens
        documents[file_name.split("\\")[0]].append(f_tokens)
        document_list.append(f_tokens)

model = Word2Vec(sentences=document_list, vector_size=16, window=5, min_count=1)

for name in documents:
    total_examples = sum(len(sentences) for sentences in documents[name])
    model.train(documents[name], total_examples=total_examples, epochs=16)

# print('wiatr i fale: ', model.wv.similarity('wiatr', 'fale'))
# print('trawie i zioła: ', model.wv.similarity('trawie', 'zioła'))
# print('zbroja i szalonych: ', model.wv.similarity('zbroja', 'szalonych'))
# print('cichym i szepotem: ', model.wv.similarity('cichym', 'szeptem'))

'''
wiatr i fale:  -0.00682733
trawie i zioła:  0.27446014
zbroja i szalonych:  -0.34637517
cichym i szepotem:  0.09730929
Największe podobieństwo słów z podanych czterech to trawie i zioła potem, trochę niższe podobieństwo to cichym i szeptem.
W przypadku kiedy wartość jest równa 0 to nie ma podobieństwa - można tak zinterpretować wiatr i fale, wartość jest bliska 0
Natomiast zbroja i szalonych to dwa słowa, które mają odwrotną zależność i wykluczają się wzajemnie z podanym wyżej prawdopodobieństwem
'''


# Zadanie 3.2
def cosine_similarity(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)


def min_max_similarity(files, matrix_similar):
    min_similarity = np.inf
    max_similarity = -np.inf
    min_pair = None
    max_pair = None

    for a, name1 in enumerate(files):
        for b, name2 in enumerate(files):
            if a != b:
                similarity = matrix_similar[a, b]
                if similarity > max_similarity:
                    max_similarity = similarity
                    max_pair = (name1, name2)
                elif similarity < min_similarity:
                    min_similarity = similarity
                    min_pair = (name1, name2)

    print(f"Most similar documents: {max_pair[0]} vs {max_pair[1]} with similarity {max_similarity:.4f}")
    print(f"Least similar documents: {min_pair[0]} vs {min_pair[1]} with similarity {min_similarity:.4f}")
    return min_pair, max_pair


documents_vector = {}

for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as f:
        vector_list = [model.wv[token] for token in filtered_tokens[file_name]]
        documents_vector[file_name] = np.mean(vector_list, axis=0)

similarity_matrix = np.zeros((len(documents_vector), len(documents_vector)))
for i, name_i in enumerate(file_names):
    for j, name_j in enumerate(file_names):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            similarity_matrix[i, j] = cosine_similarity(documents_vector[name_i], documents_vector[name_j])

# print(similarity_matrix)
# min_max_similarity(file_names, similarity_matrix)

'''
[[1.         0.64479101 0.82655591 ... 0.46966857 0.79283392 0.56877017]
 [0.64479101 1.         0.60843253 ... 0.74887562 0.8560462  0.81086326]
 [0.82655591 0.60843253 1.         ... 0.52536416 0.87624472 0.57369083]
 ...
 [0.46966857 0.74887562 0.52536416 ... 1.         0.68351948 0.9142012 ]
 [0.79283392 0.8560462  0.87624472 ... 0.68351948 1.         0.78461051]
 [0.56877017 0.81086326 0.57369083 ... 0.9142012  0.78461051 1.        ]]
Most similar documents: juliusz\juliusz_słowacki_hymn_o_zachodzie_słońca_na_morzu.txt vs juliusz\juliusz_słowacki_paryż.txt with similarity 0.9413
Least similar documents: adam\adam_mickiewicz_żal_rozrzutnika.txt vs jan\jan_kochanowski_na_dom_w_czarnolesie.txt with similarity 0.0018

Na skosie macierzy mamy liczby 1, ponieważ te same dokumenty mają prawdopodobieństwo 1:1 i nie ma sensu liczyć tego podobieństwiem cosinusowym
Jak widać podobieństwa dokumentów są różnorodne, dwa dokumenty mogą mieć podobieństwo 0.9413 czyli prawie identyczne,
 a dla innych podobieństwo wynosi 0.0018, czyli nie są w ogóle podobne do siebie
'''

# Zadanie 3.3
documents_doc2Vec = []
for file_name in file_names:
    author, doc = file_name.split("\\")
    doc = doc.split('.txt')[0]
    words = TaggedDocument(words=filtered_tokens[file_name], tags=[doc])
    documents_doc2Vec.append(words)

model_doc2Vec = Doc2Vec(documents_doc2Vec, vector_size=32, window=5)

documents_doc2Vec_vector = {}
for file_name in file_names:
    inferred_vector = model_doc2Vec.infer_vector(filtered_tokens[file_name])
    documents_doc2Vec_vector[file_name] = inferred_vector

similarity_matrix_doc2vec = np.zeros((len(documents_doc2Vec_vector), len(documents_doc2Vec_vector)))
for i, name_i in enumerate(file_names):
    for j, name_j in enumerate(file_names):
        if i == j:
            similarity_matrix_doc2vec[i, j] = 1.0
        else:
            similarity_matrix_doc2vec[i, j] = cosine_similarity(documents_doc2Vec_vector[name_i], documents_doc2Vec_vector[name_j])

print(similarity_matrix_doc2vec)
min_max_similarity(file_names, similarity_matrix_doc2vec)
