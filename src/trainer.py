import pathlib
from typing import cast
from xml.parsers.expat import model
import numpy as np
import torch
import decoder
import embedder
import json
import os

class SingleEmbeddingAxisLinearRegressionModel(torch.nn.Module):
    def __init__(self, brain_data_dim):
        """Initialize linear regression model for a single GloVe dimension."""
        super(SingleEmbeddingAxisLinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(brain_data_dim, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class SingleConceptTrainingExample:
    def __init__(self, concept: str, brain_data: np.ndarray, embedding: np.ndarray):
        self.concept = concept
        self.brain_data = brain_data  # shape: (n_features,)
        self.embedding = embedding  # shape: (300,)


class TrainingDataPerEmbeddingAxis:
    def __init__(self, concepts: list[str], brain_data: np.ndarray, embedding_values: np.ndarray):
        self.concepts = concepts
        self.brain_data_dimensions = brain_data.shape[1]
        self.brain_data = torch.autograd.Variable(torch.as_tensor(brain_data))  # shape: (n_features,)
        self.embedding_values = torch.autograd.Variable(torch.as_tensor(embedding_values))  # shape: (300,)

def train_linear_models(training_data: list[TrainingDataPerEmbeddingAxis], w1=1.0, w2=1.0):
    """
    Train ensemble of linear regression predictors for each GloVe dimension.
    
    Args:
        sessions_data: List of session data containing HBO and HBR
        glove_embeddings: GloVe embeddings (300 dimensions)
        w1, w2: Weights for HBO and HBR combination
    """
    dimension_count = 10
    
    # Store predictors for each dimension
    predictors = []
    
    for dim in range(dimension_count):
        print(f"Training predictor for dimension {dim + 1}/{dimension_count}")
        next_model = SingleEmbeddingAxisLinearRegressionModel(training_data[dim].brain_data_dimensions)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(next_model.parameters(), lr=0.01)
        predictors.append(next_model)

        for epoch in range(50):
            # Forward pass: Compute predicted y by passing 
            # x to the model
            predicted_value = next_model(training_data[dim].brain_data)

            # Compute and print loss
            loss = criterion(predicted_value, training_data[dim].embedding_values)

            # Zero gradients, perform a backward pass, 
            # and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"dim {dim}, epoch {epoch}, loss {loss.item()}")
        
    
    return predictors


def predict_glove_embeddings(predictors: list[SingleEmbeddingAxisLinearRegressionModel], brain_data: np.ndarray) -> np.ndarray:
    """
    Predict GloVe embeddings using trained ensemble.
    """
    
    predictions = np.zeros(300)
    
    for dim, model in enumerate(predictors):
        predictions[dim] = model.predict(brain_data)
    
    return predictions


def calculate_goal_feature(entry: decoder.ConceptActivationEntry, w1=1.0, w2=1.0) -> float:
    """
    Calculate combined feature for a vortex using weights w1 and w2.
    """
    return entry.hbo * w1 + entry.hbr * w2


def load_vortex_activations(sessions: list[decoder.SessionData], concepts: list[str]) -> dict[tuple[int, int, int], dict[str, float]]:
    session_ids = "-".join(str(session.idx) for session in sessions)
    cache_file = pathlib.Path(decoder.getGitRoot()) / f"data/cache/all-vortex-data-{session_ids}.json"
    
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
    os.makedirs("data/cache", exist_ok=True)
    cache_data = {}
    for coords, concept_dict in vortex_to_concept_activations.items():
        cache_data[str(coords)] = concept_dict
    
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    
    return vortex_to_concept_activations


def generate_training_data(sessions: list[decoder.SessionData], concepts: list[str], w1=1.0, w2=1.0) -> list[SingleConceptTrainingExample]:
    """
    Generate training data for ensemble predictors.
    """
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


# Example usage
if __name__ == "__main__":
    # Load your session data here
    sessions = decoder.parse_all_sessions()
    concepts = decoder.get_all_concepts(sessions)
    embedder.load_embeddings(concepts)
    concepts = list(concepts)
    
    # Train ensemble
    training_examples = generate_training_data(sessions, concepts)
    brain_data_matrix = np.array([ex.brain_data for ex in training_examples])
    training_data_per_axis: list[TrainingDataPerEmbeddingAxis] = []
    for dim in range(300):
        embedding_matrix = np.array([ex.embedding[dim] for ex in training_examples])
        training_data_per_axis.append(TrainingDataPerEmbeddingAxis(concepts, brain_data_matrix, embedding_matrix))

    predictors = train_linear_models(training_data_per_axis)
    