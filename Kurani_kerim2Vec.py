from __future__ import absolute_import, division, print_function
import codecs
import glob
import logging
import multiprocessing
import os
import pprint
import re
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
nltk.download("punkt")
nltk.download("stopwords")

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-ZŞÜÖĞÇşüöİğıç]"," ", raw)
    words = clean.split()
    return words

#book_filenames = sorted(glob.glob("meal.txt")
f = codecs.open('meal.txt', "r", encoding='utf-8')

corpus_raw = u""
for line in f:
    corpus_raw += line
    #print(line)
f.close()

tokenizer = nltk.data.load('tokenizers/punkt/turkish.pickle')

raw_sentences = tokenizer.tokenize(corpus_raw)

sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence))

print(sentences)
token_count = sum([len(sentence) for sentence in sentences])
print("The book corpus contains {0:,} tokens".format(token_count))

#ONCE we have vectors
#step 3 - build model
#3 main tasks that vectors help with
#DISTANCE, SIMILARITY, RANKING

# Dimensionality of the resulting word vectors.
#more dimensions, more computationally expensive to train
#but also more accurate
#more dimensions = more generalized
num_features = 300
# Minimum word count threshold.
min_word_count = 3

# Number of threads to run in parallel.
#more workers, faster we train
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging
seed = 1


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) &
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
        ]

    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)

meal2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

meal2vec.build_vocab(sentences)

print("Word2Vec vocabulary length:", len(meal2vec.wv.vocab))

meal2vec.train(sentences,total_examples=meal2vec.corpus_count, epochs=meal2vec.iter)

if not os.path.exists("trained"):
    os.makedirs("trained")

meal2vec.save(os.path.join("trained", "meal2vec.w2v"))

meal2vec = w2v.Word2Vec.load(os.path.join("trained", "meal2vec.w2v"))

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

all_word_vectors_matrix = meal2vec.wv.syn0

all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[meal2vec.wv.vocab[word].index])
            for word in meal2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)

print(points.head(10))

sns.set_context("poster")

print(meal2vec.most_similar("Allah"))

#points.plot.scatter("x", "y", s=10, figsize=(20, 12))

#plot_region(x_bounds=(0.0, 5.2), y_bounds=(-0.5, -0.1))
#plot_region(x_bounds=(0, 1.25), y_bounds=(0, 1.25))