import os
import numpy as np
from nltk.tokenize import word_tokenize
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

for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
        documents[file_name.split("\\")[0]].append(filtered_tokens)
        document_list.append(filtered_tokens)

model = Word2Vec(sentences=document_list, vector_size=16, window=5, min_count=1)

for name in dir_names:
    # words = [item for sublist in documents[name] for item in sublist]
    total_examples = sum(len(sentences) for sentences in documents[name])
    # print(words)
    model.train(documents[name], total_examples=total_examples, epochs=16)

# print(model.wv.key_to_index)
print(model.wv.similarity('wiatr', 'fale'))
print(model.wv.similarity('trawie', 'zioła'))
print(model.wv.similarity('zbroja', 'szalonych'))
print(model.wv.similarity('cichym', 'szeptem'))


# Zadanie 3.2
def cosine_similarity(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)


documents_vector = {}

for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
        tokens = word_tokenize(text.lower())

        filtered_tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
        vector_list = [model.wv[token] for token in filtered_tokens if token in model.wv]
        if len(vector_list) > 0:
            documents_vector[file_name] = np.mean(vector_list, axis=0)

similarity_matrix = np.zeros((len(documents_vector), len(documents_vector)))
for i, name_i in enumerate(file_names):
    for j, name_j in enumerate(file_names):
        if i == j:
            similarity_matrix[i, j] = 1.0
        else:
            similarity_matrix[i, j] = cosine_similarity(documents_vector[name_i], documents_vector[name_j])
            similarity_matrix[j, i] = similarity_matrix[i, j]


# print(similarity_matrix)
print("Similarity matrix:")
for i, name1 in enumerate(file_names):
    for j, name2 in enumerate(file_names):
        print(f"{name1} vs {name2}: {similarity_matrix[i, j]:.4f}")

max_similarity = -1
min_similarity = 2
max_pair = None
min_pair = None

for i, name1 in enumerate(file_names):
    for j, name2 in enumerate(file_names):
        if i != j:
            similarity = similarity_matrix[i, j]
            if similarity > max_similarity:
                max_similarity = similarity
                max_pair = (name1, name2)
            if similarity < min_similarity:
                min_similarity = similarity
                min_pair = (name1, name2)

print(f"Most similar documents: {max_pair[0]} vs {max_pair[1]} with similarity {max_similarity:.4f}")
print(f"Least similar documents: {min_pair[0]} vs {min_pair[1]} with similarity {min_similarity:.4f}")

# Zadanie 3.3
documents_doc2Vec = []
for file_name in file_names:
    with open(file_name, 'r', encoding='utf-8') as f:
        text = f.read()
        tokens = word_tokenize(text.lower())
        documents_doc2Vec.append(TaggedDocument(tokens, [file_name]))

model_doc2Vec = Doc2Vec(documents_doc2Vec, vector_size=32, window=5, epochs=10)

documents_doc2Vec_vector = {}
for file_name in file_names:
    author, doc = file_name.split("\\")
    inferred_vector = model_doc2Vec.infer_vector(TaggedDocument(words=doc.split('_'), tags=[file_name, author]).words)
    documents_doc2Vec_vector[file_name] = inferred_vector

max_similarity = -1
min_similarity = 2
max_pair = None
min_pair = None

for i, name1 in enumerate(file_names):
    for j, name2 in enumerate(file_names):
        if i != j:
            similarity = np.dot(documents_doc2Vec_vector[name1], documents_doc2Vec_vector[name2]) / \
                         (np.linalg.norm(documents_doc2Vec_vector[name1]) * np.linalg.norm(documents_doc2Vec_vector[name2]))
            if similarity > max_similarity:
                max_similarity = similarity
                max_pair = (name1, name2)
            if similarity < min_similarity:
                min_similarity = similarity
                min_pair = (name1, name2)

print(f"Most similar documents: {max_pair[0]} vs {max_pair[1]} with similarity {max_similarity:.4f}")
print(f"Least similar documents: {min_pair[0]} vs {min_pair[1]} with similarity {min_similarity:.4f}")