from pathlib import Path
import json
import sys
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


if __name__ == '__main__':
    sessions = decoder.parse_all_sessions()

    for (session_idx, session_data) in enumerate(sessions):
        print(session_data.stimulus_data[0])
        break

    # keys: hbo, hbr, time
    for (session_idx, session_data) in enumerate(sessions):        
        activation_entries = decoder.extract_activation_for_stimulus(session_data, session_data.stimulus_data[0])
        with open('output.txt', 'a') as f:
            f.write(f"Session: {session_idx}, concept: {session_data.stimulus_data[0].label}\n")

            for entry in activation_entries:
                f.write(f"[{entry.x}, {entry.y}, {entry.z}] - HBO: {entry.hbo}, HBR: {entry.hbr}\n")
        break

    #glove_path = 'data/glove/glove.840B.300d.txt'
    #embeddings_dict = get_embeddings_dict(glove_path)

# [time, hbo, hbr]
# hbo: []