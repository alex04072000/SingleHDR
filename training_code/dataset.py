from abc import ABC, abstractmethod
import logging
import os
import pickle
import multiprocessing as mp
import numpy as np
import cv2
cv2.setNumThreads(0)
from scipy.interpolate import interp1d

# ---

CURR_PATH_PREFIX = os.path.dirname(os.path.abspath(__file__))

np.random.seed(5)


# --- crf_list

def _get_crf_list():
    with open(os.path.join(CURR_PATH_PREFIX, 'dorfCurves.txt'), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    crf_list = [lines[idx + 5] for idx in range(0, len(lines), 6)]
    crf_list = np.float32([ele.split() for ele in crf_list])
    np.random.RandomState(730).shuffle(crf_list)
    test_crf_list = crf_list[-10:]
    train_crf_list = crf_list[:-10]

    return test_crf_list, train_crf_list


test_crf_list, train_crf_list = _get_crf_list()


# --- invcrf_list

def _inverse_rf(
        _rf,  # [s]
):
    rf = _rf.copy()
    s, = rf.shape
    rf[0] = 0.0
    rf[-1] = 1.0
    return interp1d(
        rf,
        np.linspace(0.0, 1.0, num=s),
    )(np.linspace(0.0, 1.0, num=s))


_get_invcrf_list = lambda crf_list: np.array([_inverse_rf(crf) for crf in crf_list])
test_invcrf_list = _get_invcrf_list(test_crf_list)
train_invcrf_list = _get_invcrf_list(train_crf_list)

# --- t_list

get_t_list = lambda n: 2 ** np.linspace(-3, 3, n, dtype='float32')
test_t_list = get_t_list(7)
train_t_list = get_t_list(600)


# --- Dataset, MultiDimDataset, MemDataset

class Dataset(ABC):

    # return list or np.ndarray or scalar
    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
        return DatasetIter(self)


class DatasetIter:

    def __init__(self, dataset):
        self._i = 0
        self._dataset = dataset
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._dataset):
            raise StopIteration()
        result = self._dataset[self._i]
        self._i += 1
        return result


class CatDataset(Dataset):

    def __init__(self, dataset_list):
        self._dataset_list = dataset_list
        self._len = len(dataset_list[0])
        for dataset in dataset_list:
            assert self._len == len(dataset)
        return

    def __getitem__(self, idx):
        data_list = []
        for dataset in self._dataset_list:
            data = dataset[idx]
            if type(data) is not list:
                data = [data]
            for ele in data:
                data_list.append(ele)
        return data_list

    def __len__(self):
        return self._len


class MergeDataset(Dataset):

    def __init__(self, dataset_list):
        self._dataset_list = dataset_list
        self._len = 1
        for dataset in dataset_list:
            self._len *= len(dataset)
        return

    def __getitem__(self, all_idx):
        data_list = []
        for dataset in self._dataset_list:
            all_idx, curr_idx = all_idx // len(dataset), all_idx % len(dataset)
            data = dataset[curr_idx]
            if type(data) is not list:
                data = [data]
            for ele in data:
                data_list.append(ele)
        assert all_idx == 0
        return data_list

    def __len__(self):
        return self._len


class MemDataset(Dataset):

    def __init__(self, dataset):
        self._arr = []
        for idx, ele in enumerate(dataset):
            logging.info('load dataset[%d]' % idx)
            self._arr.append(ele)
        return

    def __getitem__(self, idx):
        return self._arr[idx]

    def __len__(self):
        return len(self._arr)


# --- PatchHDRDataset

def _load_pkl(name):
    with open(os.path.join(CURR_PATH_PREFIX, name + '.pkl'), 'rb') as f:
        out = pickle.load(f)
    return out


i_dataset_train_posfix_list = _load_pkl('i_dataset_train')
i_dataset_test_posfix_list = _load_pkl('i_dataset_test')

class HDRDataset(Dataset):

    def __init__(self, hdr_prefix, hdr_posfix_list, is_training):
        self._hdr_prefix = hdr_prefix
        self._hdr_posfix_list = hdr_posfix_list
        self.is_training = is_training
        return

    def __getitem__(self, idx):
        return HDRDataset._hdr_read_resize(os.path.join(self._hdr_prefix, self._hdr_posfix_list[idx]), self.is_training)

    def __len__(self):
        return len(self._hdr_posfix_list)

    @staticmethod
    def _hdr_read(path):
        hdr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        hdr = np.flip(hdr, -1)
        hdr = np.clip(hdr, 0, None)
        return hdr

    @staticmethod
    def _hdr_resize(img, h, w):
        img = cv2.resize(img, (w, h), cv2.INTER_AREA)
        return img

    @staticmethod
    def _hdr_read_resize(path, is_training):
        hdr = HDRDataset._hdr_read(path)
        h, w, _, = hdr.shape
        ratio = max(512 / h, 512 / w)
        h = round(h * ratio)
        w = round(w * ratio)
        hdr = HDRDataset._hdr_resize(hdr, h, w)

        return hdr


class PatchHDRDataset(Dataset):

    def __init__(self, hdr_prefix, hdr_posfix_list, is_training, load_to_mem=True):
        self._hdr_dataset = HDRDataset(hdr_prefix, hdr_posfix_list, is_training)
        if load_to_mem:
            self._hdr_dataset = MemDataset(self._hdr_dataset)
        self._is_training = is_training
        return

    def __getitem__(self, idx):
        hdr = self._hdr_dataset[idx // 2]
        h, w, _, = hdr.shape
        if h > w:
            hdr = hdr[:512, :, :] if idx % 2 == 0 else hdr[-512:, :, :]
        else:
            hdr = hdr[:, :512, :] if idx % 2 == 0 else hdr[:, -512:, :]
        hdr = PatchHDRDataset._pre_hdr_p2(hdr)
        if self._is_training:
            scale = np.random.uniform(0.5, 2.0)
            hdr = cv2.resize(hdr, (np.round(512 * scale).astype(np.int32), np.round(512 * scale).astype(np.int32)), cv2.INTER_AREA)


            def randomCrop(img, width, height):
                assert img.shape[0] >= height
                assert img.shape[1] >= width
                if img.shape[1] == width or img.shape[0] == height:
                    return img
                x = np.random.randint(0, img.shape[1] - width)
                y = np.random.randint(0, img.shape[0] - height)
                img = img[y:y + height, x:x + width]
                return img

            hdr = randomCrop(hdr, 256, 256)

            hdr = np.rot90(hdr, np.random.randint(4))

            _rand_f_h = lambda: np.random.choice([True, False])
            if _rand_f_h():
                hdr = np.flip(hdr, 0)

            _rand_f_v = lambda: np.random.choice([True, False])
            if _rand_f_v():
                hdr = np.flip(hdr, 1)

        return hdr

    def __len__(self):
        return 2 * len(self._hdr_dataset)

    @staticmethod
    def _hdr_rand_flip(hdr):
        _rand_t_f = lambda: np.random.choice([True, False])
        if _rand_t_f():
            hdr = np.flip(hdr, 0)
        if _rand_t_f():
            hdr = np.flip(hdr, 1)
        return hdr

    @staticmethod
    def _pre_hdr_p2(hdr):
        hdr_mean = np.mean(hdr)
        hdr = 0.5 * hdr / (hdr_mean + 1e-6)
        return hdr


# --- get_train_dataset

def get_train_dataset(hdr_prefix):
    return MergeDataset([
        PatchHDRDataset(hdr_prefix, i_dataset_train_posfix_list, True),
        CatDataset([train_crf_list, train_invcrf_list]),
        train_t_list,
    ])


# --- get_vali_dataset
def get_vali_dataset(hdr_prefix):
    #
    posfix_list = i_dataset_test_posfix_list.copy()
    np.random.RandomState(730).shuffle(posfix_list)
    posfix_list = posfix_list[:10]

    #
    def _rand_rf_list(rf_list):
        rf_list = rf_list.copy()
        np.random.RandomState(730).shuffle(rf_list)
        rf_list = np.array(rf_list[:10])
        return rf_list

    crf_list = _rand_rf_list(test_crf_list)
    invcrf_list = _rand_rf_list(test_invcrf_list)
    #
    t_list = get_t_list(5)
    #
    return MergeDataset([
        PatchHDRDataset(hdr_prefix, posfix_list, False),
        CatDataset([crf_list, invcrf_list]),
        t_list,
    ])


# --- get_i_test_dataset, get_a_test_dataset

def get_i_test_dataset(hdr_prefix):
    return MergeDataset([
        PatchHDRDataset(hdr_prefix, i_dataset_test_posfix_list, False),
        CatDataset([test_crf_list, test_invcrf_list]),
        test_t_list,
    ])


# ---

class RandDatasetReader:

    def __init__(
            self,
            dataset,
            batch_size,
    ):
        self._n_process = 4
        self._dataset = dataset
        self._batch_size = batch_size

        # data_idx_queue
        data_idx_queue = mp.Queue(batch_size)
        enq_data_idx_p = mp.Process(
            target=RandDatasetReader._enq_data_idx,
            args=(len(dataset), data_idx_queue),
        )
        enq_data_idx_p.daemon = True
        enq_data_idx_p.start()

        # self._data_queue
        self._data_queue = mp.Queue(batch_size)
        for _ in range(self._n_process):
            p = mp.Process(
                target=RandDatasetReader._enq_data,
                args=(data_idx_queue, self._data_queue, dataset),
            )
            p.daemon = True
            p.start()

        return

    @staticmethod
    def _enq_data_idx(data_idx_max, data_idx_queue):
        while True:
            for data_idx in np.random.permutation(data_idx_max):
                data_idx_queue.put(data_idx)
        return

    @staticmethod
    def _enq_data(data_idx_queue, data_queue, dataset):
        while True:
            data_queue.put(dataset[data_idx_queue.get()])
        return

    def read_batch_data(self):
        data_list = [self._data_queue.get() for _ in range(self._batch_size)]
        _get_data_frag = lambda i: [data[i] for data in data_list]
        return [_get_data_frag(i) for i in range(len(data_list[0]))]

