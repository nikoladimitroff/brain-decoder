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
EPOCHS = 50

class BrainDecoderNN(torch.nn.Module):
    def __init__(self, brain_data_dim):
        super(BrainDecoderNN, self).__init__()
        # layer dimensions are more or less arbitrary here
        self.nn_stack = torch.nn.Sequential(
            torch.nn.Linear(brain_data_dim, 2048, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512, dtype=torch.float64),
            torch.nn.ReLU(),
            torch.nn.Linear(512, DIMENSIONS_TO_PREDICT, dtype=torch.float64),
        )

    def forward(self, x):
        y_pred = self.nn_stack(x)
        return y_pred


def train_model(training_data: list[trainer.SingleConceptTrainingExample], w1=1.0, w2=1.0) -> BrainDecoderNN:
    model = BrainDecoderNN(training_data[0].brain_data.shape[0])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(EPOCHS):
        loss = None
        for example in training_data:
            brain_data_tensor = torch.as_tensor(example.brain_data, dtype=torch.float64)
            embedding_tensor = torch.as_tensor(example.embedding, dtype=torch.float64)
            predicted_value = model(brain_data_tensor)
            loss = criterion(predicted_value, embedding_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch}, loss: {loss.item()}")

    return model


def predict_glove_embeddings(model: BrainDecoderNN, brain_data: np.ndarray) -> np.ndarray:
    brain_data_tensor = torch.as_tensor(brain_data, dtype=torch.float64)
    predicted_tensor = model(brain_data_tensor)
    return predicted_tensor.detach().numpy()


def print_predicted_concept(example: trainer.SingleConceptTrainingExample, glove_predictions):
    print(f"Concept: {example.concept}")
    print(f"Predicted: {glove_predictions}")


if __name__ == "__main__":

    print("Loading session data")
    sessions = decoder.parse_all_sessions()[0:5]
    concepts = sorted(decoder.get_all_concepts(sessions))
    embedder.load_embeddings(concepts)
    training_examples = trainer.generate_training_data(sessions, concepts)

    model_path = pathlib.Path(decoder.getGitRoot()) / "data/models/brain_decoder_nn_model.pth"
    model = BrainDecoderNN(training_examples[0].brain_data.shape[0])
    if os.path.exists(model_path):
        print("Loading trained model")
        model.load_state_dict(torch.load(model_path))
    else:        
        # Train model        
        print("Training model")
        model = train_model(training_examples)
        print("Model trained. Saving.")        
        os.makedirs(model_path.parent, exist_ok=True)
        torch.save(model.state_dict(), model_path)
    
    print("Model saved.  Loading GloVe model.")
#
    glove_path = pathlib.Path(decoder.getGitRoot()) / "data/glove/glove.840B.300d.txt"
    glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

    print("GloVe model loaded.")
    model.eval()
    for example in training_examples[15:25]:
        brain_data_tensor = torch.as_tensor(example.brain_data, dtype=torch.float64)
        sample = predict_glove_embeddings(model, brain_data_tensor)
        glove_prediction = glove_model.similar_by_vector(sample, topn=10)
        print_predicted_concept(example, glove_prediction)
        #print(f"concept: {example.concept}, difference: {[float(x - y) for x, y in zip(example.brain_data, sample)]}")
