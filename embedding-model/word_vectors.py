#!/usr/bin/env python
# Word vectors
# Gensim provides a downloader module that knows how to fetch a variety of popular pre-trained embedding and language models.
# By aliasing it as api, you get a simple interface for loading these models.
# Configure the root logger to INFO level and set a simple format
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import gensim.downloader as api
import numpy as np

model_name="glove-wiki-gigaword-100"
#model_name="word2vec-google-news-300"
logger.info(f"Loading word vectors model {model_name}.")
word_vectors = api.load(model_name)

logger.info("Word vectors loaded.")

logger.info("Look up word King:")
print( "shape: " + str(word_vectors['king'].shape))
print( "First 20 dimensions: " + str(word_vectors['king'][:20]))

# Words to visualize
words = ["king", "princess", "monarch", "throne", "crown",
         "mountain", "ocean", "tv", "rainbow", "cloud", "queen"]

# Get word vectors
vectors = np.array([word_vectors[word] for word in words])

print( "Show semantic add and subtraction: ")
analogy_vec = word_vectors['king'] + word_vectors['woman'] - word_vectors['man']
# then find the nearest word(s)
result = word_vectors.similar_by_vector(analogy_vec, topn=3)
print(result)

print("Use KeyedVectors for similarity search:")
result = word_vectors.most_similar(positive=['king','woman'],
                          negative=['man'],
                          topn=3)
print(result)
#print(type(word_vectors))

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
vectors_pca = pca.fit_transform(vectors)

# Plotting
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.scatter(vectors_pca[:, 0], vectors_pca[:, 1])
for i, word in enumerate(words):
    axes.annotate(word, (vectors_pca[i, 0]+.02, vectors_pca[i, 1]+.02))
axes.set_title('PCA of Word Embeddings')
plt.show(block=False)   # return immediately
# do whatever you like here, then…
plt.pause(5)            # keep the window open for 5 seconds
plt.close()             # close it programmatically


