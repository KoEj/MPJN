import re
import json
from wordcloud import WordCloud
from matplotlib import pyplot as plt
from os import path
import os
from PIL import Image
import numpy as np

# Zadanie 1.1
toSave = []
textFromFile = ''
with open("text.txt", encoding="utf8") as my_file:
    for line_number, line in enumerate(my_file):
        textFromFile = textFromFile + line
        if line_number % 2 == 1:
            toSave.append(line)

        # print(line_number)
        # print(line)

with open('tekst_nieparzyste.txt', 'w', encoding="utf8") as file2:
    for element in toSave:
        file2.write(element)

# Zadanie 1.2
globalWords = 0
wordsArray = []
sentences = re.split(r'(?<=\.|\?)\s', textFromFile)
globalSentences = len(sentences)
# print(sentences)

for sentence in sentences:
    # if (len(sentences) == 93):
    sent = re.sub("[\,\-\—\.]", "", sentence)
    words = sent.split()
    wordsArray.append(words)
    globalWords = globalWords + len(words)

# print(f'Words: {globalWords}')
# print(f'Sentences: {globalSentences}')
# print(sentences)
# print(wordsArray)

jsonCategories = {
    'number of words': globalWords,
    'number of sentences': globalSentences,
    'sentences': sentences,
    'words': wordsArray
}

with open('wordsSentences.json', 'w') as file:
    file.write(json.dumps(jsonCategories))

# Zadanie 1.3
wordsFromJson = None
with open('wordsSentences.json', 'r') as file:
    data = json.load(file)
    wordsFromJson = data.get('words')


res = list(map(' '.join, wordsFromJson))
s = ' '.join(str(n) for n in res)

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
masks = np.array(Image.open(path.join(d, "maska.jpg")))

wc = WordCloud(background_color="white", max_words=2000, mask=masks)
wc.generate(s)
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.savefig('chmurka.png')

stopWords = []
with open("stopwordsPL.txt", encoding="utf8") as my_file:
    for line in my_file:
        stopWords.append(line[:-1])

wc2 = WordCloud(background_color="white", max_words=2000, mask=masks, stopwords=stopWords)
wc2.generate(s)
plt.figure()
plt.imshow(wc2, interpolation="bilinear")
plt.axis("off")
plt.savefig('chmurka_lepsza.png')

