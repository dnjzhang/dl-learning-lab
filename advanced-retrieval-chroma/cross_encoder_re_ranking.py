#!/usr/bin/env python
# coding: utf-8

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
from helper_utils import load_chroma, word_wrap

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_collection = load_chroma(filename='microsoft_annual_report_2022.pdf',
                                collection_name='microsoft_annual_report_2022', embedding_function=embedding_function)
print("collection count: " + str(chroma_collection.count()))

# # Re-ranking the long tail
query = "What has been the investment in research and development?"
print("query: " + query)
results = chroma_collection.query(query_texts=query, n_results=10, include=[ 'distances', 'metadatas','documents', 'embeddings'])

retrieved_documents = results['documents'][0]

for document in results['documents'][0]:
    print((results['ids'][0]))
 #   print(str(results['metadatas'][0]))
    print(word_wrap(document[0:20]))
    print('')

from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)
print("Scores:")
for score in scores:
    print(score)


print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o+1)
