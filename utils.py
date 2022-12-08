from pathlib import Path
import numpy as np


def dirwalk(path: Path):
    """Walks through a directory and yields all file paths in the directory and subdirectories"""
    for fname in path.iterdir():
        if fname.is_dir():
            yield from dirwalk(fname)
        else:
            yield fname

def invsigmoid(x):
    return np.log(x/(1-x+1e-12))
def sigmoid(x):
    return 1/(1+np.exp(-x))