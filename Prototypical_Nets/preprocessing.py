import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm
import multiprocessing as mp
tqdm.pandas(desc="my bar!")


def image_rotate(img, angle):
    """
    Image rotation at certain angle. It is used for data augmentation
    """
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return np.expand_dims(dst, 0)


def read_alphabets(alphabet_directory, directory):
    """
    Reads all the characters from alphabet_directory and augment each image with 90, 180, 270 degrees of rotation.
    """
    datax = None
    datay = []
    characters = os.listdir(alphabet_directory)
    for character in characters:
        images = os.listdir(alphabet_directory + character + '/')
        for img in images:
            image = cv2.resize(cv2.imread(
                alphabet_directory + character + '/' + img), (28, 28))
            image90 = image_rotate(image, 90)
            image180 = image_rotate(image, 180)
            image270 = image_rotate(image, 270)
            image = np.expand_dims(image, 0)
            if datax is None:
                datax = np.vstack([image, image90, image180, image270])
            else:
                datax = np.vstack([datax, image, image90, image180, image270])
            datay.append(directory + '_' + character + '_0')
            datay.append(directory + '_' + character + '_90')
            datay.append(directory + '_' + character + '_180')
            datay.append(directory + '_' + character + '_270')
    return datax, np.array(datay)


def read_images(base_directory):
    """
    Used multithreading for data reading to decrease the reading time drastically
    """
    datax = None
    datay = []
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(read_alphabets, args=(
        base_directory + '/' + directory + '/', directory, )) for directory in os.listdir(base_directory)]
    pool.close()
    for result in results:
        if datax is None:
            datax = result[0]
            datay = result[1]
        else:
            datax = np.vstack([datax, result[0]])
            datay = np.concatenate([datay, result[1]])
    return datax, datay
