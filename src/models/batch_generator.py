import os
import random
import pickle
import numpy as np

class BatchGenerator(object):
    dataset = None
    labels = None
    translation = None

    def __init__(self, path, batch_size, seq_len, start=0, stop=-1):
        self.batch_size = batch_size
        self.seq_len = seq_len

        dataset, labels, self.translation = self.load_dataset(path, start, stop)
        ndataset, nlabels = [], []
        for i in range(len(dataset)):
            if len(dataset[i]) >= seq_len + 1:
                ndataset += [dataset[i]]
                nlabels += [labels[i]]
        del dataset, labels
        self.dataset, labels = ndataset, nlabels

        self.num_letters = len(self.translation)
        # pad all labels to be the same length
        max_len = max(map(lambda x: len(x), labels))
        self.labels = np.array([np.concatenate([np.eye(self.num_letters, dtype=np.float32)[l],
                                                np.zeros((max_len - len(l) + 1, self.num_letters),
                                                         dtype=np.float32)],
                                               axis=0)
                                for l in labels])
        self.max_len = self.labels.shape[1]
        self.indices = np.random.choice(len(self.dataset), size=(batch_size,), replace=False)
        self.batches = np.zeros((batch_size,), dtype=np.int32)

    def next_batch(self):
        coords = np.zeros((self.batch_size, self.seq_len + 1, 3), dtype=np.float32)
        sequence = np.zeros((self.batch_size, self.max_len, self.num_letters), dtype=np.float32)
        reset_states = np.ones((self.batch_size, 1), dtype=np.float32)
        needed = False
        for i in range(self.batch_size):
            if self.batches[i] + self.seq_len + 1 > self.dataset[self.indices[i]].shape[0]:
                ni = random.randint(0, len(self.dataset) - 1)
                self.indices[i] = ni
                self.batches[i] = 0
                reset_states[i] = 0.
                needed = True
            coords[i, :, :] = self.dataset[self.indices[i]][self.batches[i]: self.batches[i] + self.seq_len + 1]
            sequence[i] = self.labels[self.indices[i]]
            self.batches[i] += self.seq_len

        return coords, sequence, reset_states, needed

    def reset(self):
        self.indices = np.random.choice(len(self.dataset), size=(self.batch_size,), replace=False)
        self.batches = np.zeros((self.batch_size,), dtype=np.int32)

    @staticmethod
    def load_dataset(path, start=0, stop=-1):
        if BatchGenerator.dataset is None:
            BatchGenerator.dataset = np.load(os.path.join(path, 'dataset.npy'))

        dataset = BatchGenerator.dataset
        dataset = [np.array(d) for d in dataset]
        if stop == -1:
            stop = len(dataset)

        dataset = dataset[start:stop]
        temp = []
        for d in dataset:
            # dataset stores actual pen points, but we will train on differences between consecutive points
            offs = d[1:, :2] - d[:-1, :2]
            ends = d[1:, 2]
            temp += [np.concatenate([[[0., 0., 1.]], np.concatenate([offs, ends[:, None]], axis=1)], axis=0)]
        # because lines are of different length, we store them in python array (not numpy)
        dataset = temp

        if BatchGenerator.labels is None:
            BatchGenerator.labels = np.load(os.path.join(path, 'labels.npy'))[start:stop]
        labels = BatchGenerator.labels

        if BatchGenerator.translation is None:
            BatchGenerator.translation = np.load(os.path.join(path, 'translation'))
        translation = BatchGenerator.translation

        return dataset, labels, translation

    @staticmethod
    def dataset_size(path):
        if BatchGenerator.dataset is None:
            BatchGenerator.dataset = np.load(os.path.join(path, 'dataset.npy'))

        dataset = BatchGenerator.dataset
        l = len(dataset)
        return l

