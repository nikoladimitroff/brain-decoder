
notes

assumptions: I use 'neurons firing' or 'neurons activating' as a proxy for 'voxels with significant HBO/HBR values'

1. data needed to be masked - gazillion 0s
2. concepts are not unique within a session - some concepts appear more than once, some don't appear in all sessions
3. hbo / hbr values are sometimes all over the place; there's clearly lots of noise depending
4. should probably take the difference between the rest period and the actual stimulus
5. lots of neurons seem to activate all the time - boost larger activations by raising the stimulus to ^2
6. plots now seem significantly different for the different words
7. i should probably take into account the positioning and type of the core words around the main concept but for this particular experiment I will ignore them
8. Next step: learning a model from the current word totals
9. Types of models I am considering:
    - Train a single predictor (doesn't matter what kind exactly, but let's say a NN) that takes as input activated neurons and returns a vector in the GloVe embedding space, then take closest neighbour
        - pros: just a single model, can detect more complex patterns, more likely to overfit
        - cons: costlier to train (especially in a home environment)
    - Train an ensemble of predictors (again, doesn't matter what kind), one per axis in the GloVe embedding space, that focus on predicting how neurons activating correspond to changes in each axis
        - pros: can be trained one at a time (saves development time), less overfitting,
        clear path to interpretability - for every axis, plot every concept's embedding value , plot how neurons predict 
        - cons: assumes the axis in the embedding space are linearly independent,
10. found similar sets in keggel