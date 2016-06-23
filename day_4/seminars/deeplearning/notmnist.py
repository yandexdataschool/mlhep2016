"""
Taken from https://gist.github.com/Dorian
"""

import os
import sys
import tarfile
import urllib

import numpy as np
from scipy.misc import imread

url = 'http://yaroslavvb.com/upload/notMNIST/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print 'Found and verified', filename
    else:
        raise Exception(
            'Failed to verify' + filename + '. Can you get to it with a browser?')
    sys.stdout.flush()
    return filename


num_classes = 10


def extract(filename):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if not os.path.exists(root):
        print 'extracting...'
        tar = tarfile.open(filename)
        tar.extractall()
        tar.close()
    else:
        print 'using cached folder...'
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root))]
    if len(data_folders) != num_classes:
        raise Exception('Expected %d folders, one per class. Found %d instead.' % (num_classes, len(data_folders)))
    sys.stdout.flush()
    return data_folders


image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.


def load(data_folders, min_num_images, max_num_images):
    dataset = np.ndarray(
        shape=(max_num_images, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
    label_index = 0
    image_index = 0
    for folder in data_folders:
        print folder
        for image in os.listdir(folder):
            if image_index >= max_num_images:
                raise Exception('More images than expected: %d >= %d' % (image_index, max_num_images))
            image_file = os.path.join(folder, image)
            try:
                image_data = (imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[image_index, :, :] = image_data
                labels[image_index] = label_index
                image_index += 1
            except IOError as e:
                print 'Could not read:', image_file, ':', e, '- it\'s ok, skipping.'
                label_index += 1
    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    labels = labels[0:num_images]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (
            num_images, min_num_images))
    print 'Full dataset tensor:', dataset.shape
    print 'Mean:', np.mean(dataset)
    print 'Standard deviation:', np.std(dataset)
    print 'Labels:', labels.shape
    sys.stdout.flush()
    return dataset, labels


from sklearn.cross_validation import train_test_split


def load_dataset(test_size=0.2):
    if not os.path.exists('data.npz'):
        print "Downloading and preprocessing dataset. This may take a few minutes"

        train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
        test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
        train_folders = extract(train_filename)
        test_folders = extract(test_filename)

        X, y = load(train_folders, 450000, 550000)
        X = X[:, None, :, :]
        np.random.seed(1337)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
        X_test, y_test = load(test_folders, 18000, 20000)
        X_test = X_test[:, None, :, :]

        with open('data.npz', 'w') as fout:
            np.savez_compressed('data.npz', *[X_train, y_train, X_val, y_val, X_test, y_test])

    else:
        print 'using stored data.npz'
        sys.stdout.flush()
        [X_train, y_train, X_val, y_val, X_test, y_test] = map(np.load('data.npz').__getitem__,
                                                               map("arr_{}".format, range(6)))
    return X_train, y_train, X_val, y_val, X_test, y_test
