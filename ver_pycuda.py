import numpy as np

import random
import datetime

from pycuda_func import parallel_neighbor_search

# designed to work with mnist 28*28 hand written digits

data_size = 784
K = 10
split_ratio = 0.2  # validation / total
folds = 5
performance_metrics = "./summary_gpu.txt"


def main():

    # read training data
    print(datetime.datetime.now())
    all_training = read_csv("./data/train.csv")[1:]

    data_len = len(all_training)
    val_num = int(split_ratio * data_len)
    train_num = data_len - val_num

    master_confusion_matrix = np.tile(np.zeros(10), (10, 1))

    # CV loop
    for fold in range(folds):
        print("\nFold " + str(fold) + "\n")
        confusion_matrix = np.tile(np.zeros(10), (10, 1))

        indexes = np.array(range(data_len))
        random.Random(fold).shuffle(indexes)

        neighbors = [all_training[train] for train in indexes[:train_num]]
        validation = [all_training[valid] for valid in indexes[train_num:]]

    # cross validation, split

        val_digits = np.array([item[0] for item in validation])
        subjects = np.array([item[1:] for item in validation])

        predictions = nearest_neighbors_search(
            subjects, neighbors, val_num, train_num, K=10)

        for i, item in enumerate(predictions):
            confusion_matrix[val_digits[i]][item] += 1

    # test loop

    # performance metrics

        with open(performance_metrics, 'a') as f:
            f.write("Fold " + str(fold) + ":\n=====================\n")
            f.write("\\ " + str(range(10)) +
                    "\n \\--------------------------------------\n")
            for i, line in enumerate(confusion_matrix):
                f.write(str(i) + "|" + str(line) + "\n")
            f.write("\n")
        master_confusion_matrix += confusion_matrix
    # read test data

    with open(performance_metrics, 'a') as f:
        f.write("Master:\n=====================")
        f.write("\\ " + str(range(10)) +
                "\n \\--------------------------------------\n")
        for i, line in enumerate(master_confusion_matrix):
            f.write(str(i) + "|" + str(line) + "\n")
        f.write("\n")
    # inference


def read_csv(path: str):
    return np.genfromtxt(path, delimiter=',', dtype=np.uint8)


def nearest_neighbors_search(subjects, neighbors, val_num, train_num, K: int = K,) -> int:

    return parallel_neighbor_search(neighbors, subjects, val_num, train_num, K)


if __name__ == "__main__":
    main()
    # void()
