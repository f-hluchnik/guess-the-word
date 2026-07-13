import os
import urllib.request
import numpy as np
from gensim.models import KeyedVectors

VECTOR_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.cs.300.vec.gz"
VECTOR_PATH = "cc.cs.300.vec.gz"
VECTOR_PATH_KV = "cc.cs.300.kv"

# 1. LOAD A PRE-TRAINED CZECH VECTOR FILE (Loaded once into RAM on server start)
# This example uses Facebook's open-source fastText crawl vectors for Czech
# URL: https://fasttext.cc/docs/en/crawl-vectors.html
if not os.path.exists(VECTOR_PATH):
    print(f"Downloading Czech vectors from {VECTOR_URL} ...")
    urllib.request.urlretrieve(VECTOR_URL, VECTOR_PATH)

if not os.path.exists(VECTOR_PATH_KV):
    print(f"Converting Czech vectors to KeyedVectors format...")
    word_vectors = KeyedVectors.load_word2vec_format(VECTOR_PATH, binary=False, limit=500000)
    word_vectors.save(VECTOR_PATH_KV)

print("Loading Czech semantic database...")
word_vectors = KeyedVectors.load(VECTOR_PATH_KV)

def get_word_similarity(guess_word, target_word):
    # Ensure words exist in our mathematical universe
    if guess_word not in word_vectors or target_word not in word_vectors:
        return "Word not found in the dictionary."
    
    # 2. CALCULATE THE COSINE SIMILARITY (Pure vector math, takes ~0.0001 seconds)
    similarity = word_vectors.similarity(guess_word, target_word)
    
    temperature = max(0.0, float(similarity) * 100)
    return round(temperature, 1)

# --- EXAMPLE LIVE TEST ---
target = "víla"
print(f"skřítek -> {get_word_similarity('skřítek', target)} °C")
print(f"les -> {get_word_similarity('les', target)} °C")
print(f"lesomílek -> {get_word_similarity('lesomílek', target)}")