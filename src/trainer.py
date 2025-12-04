from multiprocessing.pool import Pool
import pathlib
import json
import os
from typing import cast

import numpy as np

import decoder
import embedder

DIMENSIONS_TO_PREDICT = 300

class SingleConceptTrainingExample:
    def __init__(self, concept: str, brain_data: np.ndarray, embedding: np.ndarray):
        self.concept = concept
        self.brain_data = brain_data  # shape: (n_features,)
        self.embedding = embedding  # shape: (300,)

def calculate_goal_feature(entry: decoder.ConceptActivationEntry, w1=2.0, w2=1.0) -> float:
    """
    Calculate combined feature for a vortex using weights w1 and w2 - this makes the implementation
    easier because we don't have to deal with multi-dimensional outputs in the model.
    The values are completely arbitrary and can be tuned later.
    HBO is weighted more heavily because it tends to have stronger corelation
    with brain activation according to some papers.
    """
    return entry.hbo * w1 + entry.hbr * w2


def load_vortex_activations(sessions: list[decoder.SessionData], concepts: list[str]) -> dict[tuple[int, int, int], dict[str, float]]:
    session_ids = "-".join(str(session.idx) for session in sessions)
    cache_file = pathlib.Path(decoder.getGitRoot()) / f"data/cache/vortex-data/{len(concepts)}-vortex-data-({session_ids}).json"
    
    vortex_to_concept_activations: dict[tuple[int, int, int], dict[str, float]] = {}

    # If the cache file exists, load from it
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        # Convert string keys back to tuples
        vortex_to_concept_activations = {}
        for coords_str, concept_dict in cached_data.items():
            coords = cast(tuple[int, int, int], tuple(map(int, coords_str.strip('()').split(', '))))
            vortex_to_concept_activations[coords] = concept_dict
        return vortex_to_concept_activations
    
    # Process data as before...
    for concept in concepts:
        brain_data = decoder.extract_average_stimulus_delta(sessions, concept)
        for entry in brain_data:
            coords = (entry.x, entry.y, entry.z)
            if vortex_to_concept_activations.get(coords) is None:
                vortex_to_concept_activations[coords] = {}
            vortex_to_concept_activations[coords][concept] = calculate_goal_feature(entry)
    
    # Save to cache
    os.makedirs("data/cache/vortex-data", exist_ok=True)
    cache_data = {}
    for coords, concept_dict in vortex_to_concept_activations.items():
        cache_data[str(coords)] = concept_dict
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    
    return vortex_to_concept_activations


def generate_training_data(sessions: list[decoder.SessionData], concepts: list[str], w1=1.0, w2=1.0) -> list[SingleConceptTrainingExample]:
    # Step 1: gather all vortices that have non-zero activations for at least 1 concept to reduce data size
    vortex_to_concept_activations: dict[tuple[int, int, int], dict[str, float]] = load_vortex_activations(sessions, concepts)

    active_vortex_coordinates = list(vortex_to_concept_activations.keys())
    active_vortex_coordinates.sort()

    # Step 2: create training examples - 1 feature vector per concept
    all_examples: list[SingleConceptTrainingExample] = []
    for concept in concepts:
        # Can't use a complete vortex as the dict doesnt guarantee key iteration order - use sorted list instead
        activation_values = np.zeros(len(active_vortex_coordinates))
        for i, coords in enumerate(active_vortex_coordinates):
            activation_values[i] = vortex_to_concept_activations[coords].get(concept, 0.0)
        glove_vector = embedder.get_glove_embedding(concept)
        assert glove_vector is not None, f"No GloVe embedding found for concept '{concept}'"
        all_examples.append(SingleConceptTrainingExample(concept, activation_values, glove_vector))

    return all_examples

