import numpy as np
import datetime
import random


# designed to work with mnist 28*28 hand written digits

data_size = "auto"
K = 10
split_ratio = 0.2  # validation / total
folds = 5
performance_metrics = "./summary.txt"


def main():

    # read training data
    print(datetime.datetime.now())
    all_training = read_csv("./data/small.csv")[1:]

    data_len = len(all_training)
    val_num = int(split_ratio * data_len)
    train_num = data_len - val_num

    master_confusion_matrix = np.tile(np.zeros(10), (10, 1))

    # CV loop
    for fold in range(folds):
        print("\nFold " + str(fold))
        confusion_matrix = np.tile(np.zeros(10), (10, 1))

        indexes = np.array(range(data_len))
        # random.Random(fold).shuffle(indexes)

        neighbors = [all_training[train] for train in indexes[:train_num]]
        validation = [all_training[valid] for valid in indexes[train_num:]]

    # cross validation, split
        counter = 0
        big_counter = 0
        for item in validation:
            counter += 1
            confusion_matrix[item[0]][nearest_neighbors_search(
                item, neighbors)] += 1
            if (counter == val_num / 100):
                big_counter += 1
                counter = 0
                print("Progress on fold " + str(fold) + ": %" +
                      str(big_counter) + ", " + str(datetime.datetime.now()), end="\r", flush=True)

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


def nearest_neighbors_search(subject, neighbors, K: int = K,) -> int:

    dist_array = np.array([np.linalg.norm(subject[1:] - other[1:])
                           for other in neighbors])
    index_arr = np.argpartition(dist_array, K)[:K]
    return most_frequent([item[0] for item in [neighbors[p] for p in index_arr]])


def most_frequent(List: list) -> int:
    return max(set(List), key=List.count)


if __name__ == "__main__":
    main()
