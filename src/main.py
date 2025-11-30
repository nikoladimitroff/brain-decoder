from pathlib import Path
import json
import sys
import time
import numpy as np
import os
import subprocess

from entries import *
import decoder

def main():
    
    sessions = decoder.parse_all_sessions()

    start_time = time.time() # Record start time

    activations = decoder.extract_average_stimulus_delta(sessions, "land")
    # keys: hbo, hbr, time
    f = open("test_2.txt", 'a')
    for (session_idx, session_data) in enumerate(sessions):
        
        #s = session_data.stimulus_data[0]
        print(",".join(s.label for s in session_data.stimulus_data), file=f)
        #activations = decoder.extract_activations_for_stimulus(session_data, s)
        #
        #with open("test_2.json", 'a') as f:
        #    json.dump([entry.to_dict() for entry in activations], f)
        #break
    f.close()
    end_time = time.time() # Record end time
    print(f"Took {end_time - start_time:.2f} seconds to process sessions.")

    #glove_path = 'data/glove/glove.840B.300d.txt'
    #embeddings_dict = get_embeddings_dict(glove_path)

# [time, hbo, hbr]
# hbo: []

if __name__ == '__main__':
    main()