

import numpy as np
from pathlib import Path
from decoder import getGitRoot

GLOVE_EMBEDDINGS = {}

def get_glove_embedding(stimulus):
    return GLOVE_EMBEDDINGS[stimulus]    

def load_embeddings(concepts: set[str], embedding_dim=300):
    """
    Download the GloVe embeddings from:
        wget http://nlp.stanford.edu/data/glove.840B.300d.zip
        unzip glove.840B.300d.zip
    """
    glove_path = Path(getGitRoot()) / 'data/glove/glove.840B.300d.txt'
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Don't split the whole line before we know we want to keep it
            first_space = line.find(" ")
            word = line[:first_space]
            if word not in concepts:
                continue
            values = line.rstrip().rsplit(' ', embedding_dim)
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    global GLOVE_EMBEDDINGS
    GLOVE_EMBEDDINGS = embeddings

