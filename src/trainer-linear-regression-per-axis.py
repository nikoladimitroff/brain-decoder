from multiprocessing.pool import Pool
import pathlib
import json
import os
from typing import cast

import numpy as np
import torch
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import decoder
import embedder
import trainer

DIMENSIONS_TO_PREDICT = 300

class SingleEmbeddingAxisLinearRegressionModel(torch.nn.Module):
    def __init__(self, brain_data_dim):
        """Initialize linear regression model for a single GloVe dimension."""
        super(SingleEmbeddingAxisLinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(brain_data_dim, 1, bias=True, dtype=torch.float64)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class TrainingDataPerEmbeddingAxis:
    def __init__(self, concepts: list[str], brain_data: np.ndarray, embedding_values: np.ndarray):
        self.concepts = concepts
        self.brain_data_dimensions = brain_data.shape[1]
        self.brain_data = torch.autograd.Variable(torch.as_tensor(brain_data, dtype=torch.float64))  # shape: (n_features,)
        self.embedding_values = torch.autograd.Variable(torch.as_tensor(embedding_values, dtype=torch.float64))  # shape: (300,)


def train_single_axis_model(args: tuple[int, TrainingDataPerEmbeddingAxis]) -> SingleEmbeddingAxisLinearRegressionModel:
    dim, training_data = args
    print(f"Training predictor for dimension {dim + 1}/{DIMENSIONS_TO_PREDICT}")
    next_model = SingleEmbeddingAxisLinearRegressionModel(training_data.brain_data_dimensions)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(next_model.parameters(), lr=0.01)
    
    loss = None
    for epoch in range(250):
        predicted_value = next_model(training_data.brain_data)
        loss = criterion(predicted_value, training_data.embedding_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Dim {dim}, final loss: {loss.item()}")
    return next_model


def train_linear_models(training_data: list[TrainingDataPerEmbeddingAxis], w1=1.0, w2=1.0) -> list[SingleEmbeddingAxisLinearRegressionModel]:
    """
    Train ensemble of linear regression predictors for each GloVe dimension.
    
    Args:
        sessions_data: List of session data containing HBO and HBR
        glove_embeddings: GloVe embeddings (300 dimensions)
        w1, w2: Weights for HBO and HBR combination
    """
    
    # Store predictors for each dimension
    predictors = []
    
    with Pool(16) as p:
        predictors = p.map(train_single_axis_model, [(dim, training_data[dim]) for dim in range(DIMENSIONS_TO_PREDICT)])
    
    return predictors


def predict_glove_embeddings(predictors: list[SingleEmbeddingAxisLinearRegressionModel], brain_data: np.ndarray) -> np.ndarray:
    """
    Predict GloVe embeddings using trained ensemble.
    """
    
    predictions = np.zeros(DIMENSIONS_TO_PREDICT)
    
    for dim, model in enumerate(predictors):
        predictions[dim] = model(brain_data)
    
    return predictions


def print_predicted_concept(example: trainer.SingleConceptTrainingExample, glove_predictions):
    print(f"Concept: {example.concept}")
    print(f"Predicted: {glove_predictions}")

if __name__ == "__main__":

    # Load session data
    sessions = decoder.parse_all_sessions()[0:5]
    concepts = sorted(decoder.get_all_concepts(sessions))[0:60]
    embedder.load_embeddings(concepts)
    
    # Train all predictors for each GloVe dimension
    
    training_examples = trainer.generate_training_data(sessions, concepts)
    brain_data_matrix = np.array([ex.brain_data for ex in training_examples])
    training_data_per_axis: list[TrainingDataPerEmbeddingAxis] = []

    for axis_idx in range(DIMENSIONS_TO_PREDICT):
        embedding_values = np.array([ex.embedding[axis_idx] for ex in training_examples])
        training_data_per_axis.append(TrainingDataPerEmbeddingAxis(concepts, brain_data_matrix, embedding_values))

    print("Training models")
    predictors = train_linear_models(training_data_per_axis)

    print("Models trained.  Loading GloVe model.")
    
    glove_path = pathlib.Path(decoder.getGitRoot()) / 'data/glove/glove.840B.300d.txt'
    glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

    print("GloVe model loaded.")


    for example in training_examples[15:30]:
        brain_data_tensor = torch.as_tensor(example.brain_data, dtype=torch.float64)
        sample = predict_glove_embeddings(predictors, brain_data_tensor)
        glove_prediction = glove_model.similar_by_vector(sample, topn=10)
        print_predicted_concept(example, glove_prediction)
    