#### Load package ####
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import pickle
import os
import tifffile

Image.MAX_IMAGE_PIXELS = None
PIL.Image.MAX_IMAGE_PIXELS = 10e100
import tracemalloc
from functools import wraps

def measure_peak_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        current_gb = current / (1024 ** 3)
        peak_gb = peak / (1024 ** 3)
        print(f"[{func.__name__}] Current memory: {current_gb:.4f} GB; Peak memory: {peak_gb:.4f} GB")
        tracemalloc.stop()
        return result
    return wrapper


#### Basic functions ####

def mkdir(path):
    dirname = os.path.dirname(path)
    if dirname != '':
        os.makedirs(dirname, exist_ok=True)


def get_image_filename(prefix):
    file_exists = False
    for suffix in ['.jpg', '.png', '.tiff', '.tif']:
        filename = prefix + suffix
        if os.path.exists(filename):
            file_exists = True
            break
    if not file_exists:
        raise FileNotFoundError('Image not found')
    return filename


def smart_save_image(img, prefix, base_name="base", size_threshold=60000):
    """
    Save image as JPG if both dimensions are under `size_threshold`, otherwise as TIFF.
    """
    h, w = img.shape[:2]
    print(f"Image size: {h}x{w}")

    if h < size_threshold and w < size_threshold:
        # Save as JPG
        path = f"{prefix}{base_name}.jpg"
        Image.fromarray(img.astype(np.uint8)).save(path, quality=90)
        print(f"✅ Saved as JPG: {path}")
    else:
        # Save as TIFF
        path = f"{prefix}{base_name}.tiff"
        tifffile.imwrite(path, img, bigtiff=True)
        print(f"✅ Saved as TIFF: {path}")




def read_string(filename):
    return read_lines(filename)[0]


def write_string(string, filename):
    return write_lines([string], filename)


def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img


def save_image(img, filename):
    mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.rstrip() for line in file]
    return lines


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def load_tsv(filename, index=True):
    if index:
        index_col = 0
    else:
        index_col = None
    df = pd.read_csv(filename, sep='\t', header=0, index_col=index_col)
    print(f'Dataframe loaded from {filename}')
    return df


def save_tsv(x, filename, **kwargs):
    mkdir(filename)
    if 'sep' not in kwargs.keys():
        kwargs['sep'] = '\t'
    x.to_csv(filename, **kwargs)
    print(filename)

        
def save_pickle(x, filename):
    mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)


def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask

    


