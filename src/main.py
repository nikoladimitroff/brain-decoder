from pathlib import Path
import json
import sys
import time
import numpy as np
import os
import subprocess

from entries import *
import decoder

def get_glove_embedding(stimulus, embeddings_dict):
    return embeddings_dict[stimulus]
    

def get_embeddings_dict(glove_path, embedding_dim=300):
    """
    Download the GloVe embeddings from:
        wget http://nlp.stanford.edu/data/glove.840B.300d.zip
        unzip glove.840B.300d.zip
    """
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ', embedding_dim)
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def main():
    
    sessions = decoder.parse_all_sessions()

    start_time = time.time() # Record start time
    # keys: hbo, hbr, time
    for (session_idx, session_data) in enumerate(sessions):
        
        s = session_data.stimulus_data[0]
        activations = decoder.extract_activations_for_stimulus_2(session_data, s)
        
        with open("test_2.json", 'a') as f:
            json.dump([entry.to_dict() for entry in activations], f)
        break

    end_time = time.time() # Record end time
    print(f"Took {end_time - start_time:.2f} seconds to process sessions.")

    #glove_path = 'data/glove/glove.840B.300d.txt'
    #embeddings_dict = get_embeddings_dict(glove_path)

# [time, hbo, hbr]
# hbo: []

if __name__ == '__main__':
    main()