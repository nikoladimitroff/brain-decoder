from pathlib import Path
import json
import sys
import numpy as np
import decoder
import embedder
from multiprocessing import Pool

sessions = decoder.parse_all_sessions()[0:5]

def extract_stimuli(concept):
    brain_data = decoder.extract_average_stimulus_delta(sessions, concept)

# Example usage
if __name__ == "__main__":
    # Load your session data here
    concepts = decoder.get_all_concepts(sessions)
    concepts = sorted(list(concepts))
    
    with Pool(16) as p:
        p.map(extract_stimuli, concepts)
    