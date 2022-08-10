import os, glob
import numpy as np
import sys
from joblib import Parallel, delayed, parallel
import contextlib
from tqdm import tqdm
import multiprocessing

from modules.getFeatures import get_features

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

languages = {
    'de':'German',
    'en':'English',
    'es':'Spanish'
}

def getInputs(directory):
    return glob.glob(directory)

def processInput(file):
    file_name = os.path.basename(file)
    language = languages[file_name.split("_")[0]]
    features = get_features(file)
    return features, language

def create_data(directory):
    inputs = getInputs(directory)
    num_cores = multiprocessing.cpu_count()
    with tqdm_joblib(tqdm(desc="Testing models", total=len(inputs))) as progress_bar:
        results = Parallel(n_jobs=num_cores)(delayed(processInput)(file) for file in inputs)
    features, languages = zip(*results)
    return np.array(features), np.array(languages)

#features, languages = create_data("dataset/archive(1)/train/train/*.flac")
#np.save('train-features-4.npy', features)
#np.save('train-languages-4.npy', languages)

def main():
    features, languages = create_data(sys.argv[1])
    np.save(f'{sys.argv[2]}-features.npy', features)
    np.save(f'{sys.argv[2]}-languages.npy', languages)

if __name__ == "__main__":
    main()