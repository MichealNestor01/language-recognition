import numpy as np

def loadTrainingData(filename):
    features = np.load(f'./{filename}-features.npy', allow_pickle=True)
    languages = np.load(f'./{filename}-languages.npy', allow_pickle=True)
    return features, languages

def loadTestData(filename):
    features = np.load(f'./{filename}-features.npy', allow_pickle=True)
    languages = np.load(f'./{filename}-languages.npy', allow_pickle=True)
    return features, languages    