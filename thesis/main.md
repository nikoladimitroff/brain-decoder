
notes

assumptions:

1. I use 'neurons firing' or 'neurons activating' as a proxy for 'voxels with significant HBO/HBR values' as that's what Wikipedia promises in this article: https://en.wikipedia.org/wiki/Neurovascular_unit
2. This paper says HBO and HBR are inversely correlated about the corelation between HBO and HBR: https://www.mdpi.com/1424-8220/23/8/3979
3. Multiple papers talk about how accurate fNIRS measurements need to be broken down to short and long channels (e.g. https://pmc.ncbi.nlm.nih.gov/articles/PMC7757903/). It's unclear how the current dataset was measured so it might be affected by extra noise.
4. This paper talks about massive gender/age differences in the recorded fNIRS data: https://www.researchgate.net/publication/326962527_Impact_of_Healthy_Aging_on_Multifractal_Hemodynamic_Fluctuations_in_the_Human_Prefrontal_Cortex. This means any actual brain decoder would have to isolate all of that noise.
5. This website seems to give a good summary of fNIRS data analysis but I don't have time to read it all - probably a good idea: https://pvrticka.com/fnirs-hyperscanning-an-introduction/. Key information I'm using from it:
    - HBO is more closely corelated with the actual neural excitement than HBR. I use this for training purposes to define a target feature for our training algorithm. This is further confirmed by our data as well - my plots of the HBO vs HBR show much bigger HBO effect for a random collection of concepts.
    - Task duration is critical - that website and also some other papers I lost the links to, discuss that ideal duration should be in the 15-30s range (even the haemodynamic response curve from Wikipedia is drawn over 15s with the peak response arriving about 5s afterwards). This is quite worrying as most of the stimuli in our sample data takes only 3s but that's one more thing to ignore.


1. data needed to be masked - gazillion 0s
2. concepts are not unique within a session - some concepts appear more than once, some don't appear in all sessions
3. hbo / hbr values are sometimes all over the place; there's clearly lots of noise
4. should take the difference between the rest period and the actual stimulus to find out what the hemodynamic response is
    4.1 The hemodynamic response curve starts strong and then quickly decays.
    [!images/external/haemodynamic-curve.png] This means we are likely interested in taking just the amplitude of the result. This is further talked about in [this paper](https://www.sciencedirect.com/science/article/pii/S266695602200023X) which confirms HBO/HBR's amplitude can be used as a proxy for neural activity.
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
        clear path to interpretability - for every axis, plot every concept's embedding value, plot how neurons predict each axis
        - cons: assumes the axis in the embedding space are linearly independent,
10. Feature engineering:
    - lots of 0s, need to only include non-zero vortices
    - papers suggest that HBO is more closely corelated to the actual neural activation. For that reason, and for a lack of deeper neuroscience knowledge, and for the sake of designing a simpler predictor for the experiment, I'll use a target feature that uses HBO with a higher weight than HBR.
10. found similar sets in keggel

## Ideas to explore if I had infinite time:

1. Change the basis of the GloVe embeddings into dimensions that actually have assigned semantics e.g.
    - embeddings[0] to signify gender (`dot(king-queen, w) for w in glove`)
    - embeddings[1] to signify goodness (`dot(good-bad, w) for w in glove`)
    - etc.
This would provide much greater level of interpretability of any model being trained 

2. Explore treating concepts like roots in Arabic. Arabic and other Semitic languages don't form words with prefixes + root + suffix but instead through pattern matching. Here's an axamples for the root k-t-b for "something related to writing"
    - k-t-b (root) + CaCaaC (pattern for object) = kitaab = book i.e. "writing" + "object" = "book"
    - k-t-b (root) + CaCiC (pattern for subject) = katib = writer i.e. "writing" + "subject" = "writer'
    - k-t-b (root) + maCCaC (pattern for place of action) = maktab = desk i.e. "writing" + "place of action" = "desk"

There have been medical studies confirming that Arabic speakers actually form words in 2-dimensions - "what the word relates to" vs "what information the word passes". What if everyone actually forms words in multiple dimensions but we just haven't realized that and we can actually map different brain regions to 'roots' and 'patterns'?

3. Eradicate noise from the dataset:
    - is there a difference between the same person at different times of the day? How about different coffeine or alcohol levels? Other factors?
    - is there a difference between different language speakers? (goes back to the Arabic)
    - how big is the measured data difference because of an age or gender difference? How to normalize it?
    - map brain regions - are certain regions more excited for more certain concepts? what does neuroscience say about it?
    - are all neural activations equally important? Can we reduce the feature count by eliminating low-variance dimensions or dimensions that are noisy?