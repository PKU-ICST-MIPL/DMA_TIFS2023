from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import glob
import re

class sysu(object):
    def __init__(self):
        self.dataset_dir = '/home/cuizhenyu/Dataset_VIReID/SYSU-MM01/'
        self.train_rgb_dir = osp.join(self.dataset_dir, 'train_rgb')
        self.gallery_rgb_dir = osp.join(self.dataset_dir, 'gallery_rgb')
        self.train_ir_dir = osp.join(self.dataset_dir, 'train_ir')
        self.query_ir_dir = osp.join(self.dataset_dir, 'query_ir')

        train_rgb, num_train_pids = self._process_dir(self.train_rgb_dir, relabel=True)
        train_ir, _ = self._process_dir(self.train_ir_dir, relabel=True)
        query_ir, num_test_pids = self._process_dir(self.query_ir_dir)
        gallery_rgb, _ = self._process_dir(self.gallery_rgb_dir)

        print("=> sysu loaded")
        print("  --------------------------------")
        print("  subset       | # ids | # images")
        print("  --------------------------------")
        print("  train_rgb    | {:5d} | {:8d}".format(num_train_pids, len(train_rgb)))
        print("  train_ir     | {:5d} | {:8d}".format(num_train_pids, len(train_ir)))
        print("  query_ir     | {:5d} | {:8d}".format(num_test_pids, len(query_ir)))
        print("  gallery_rgb  | {:5d} | {:8d}".format(num_test_pids, len(gallery_rgb)))
        print("  --------------------------------")
        self.train = train_rgb + train_ir
        self.query = query_ir
        self.gallery = gallery_rgb
        self.num_train_pids = num_train_pids

        #########################################################################
        # self.qg = self._process_dir(self.gallery_rgb_dir)[0] + self._process_dir(self.query_ir_dir)[0]

    def _process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, '*.[jp][pn]g')))
        pattern = re.compile(r'(\d+)_c(\d)')
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        num_pids = len(pid_container)
        return dataset, num_pids

class regdb(object):
    def __init__(self, trial = 1):
        self.dataset_dir = '/home/cuizhenyu/Dataset_VIReID/RegDB/'
        train_rgb_list = self.dataset_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_ir_list = self.dataset_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'
        test_rgb_list = self.dataset_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
        test_ir_list = self.dataset_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
        rgb_train, num_train_pids, rgb_train_imgs = \
            self._process_dir(self.dataset_dir, train_rgb_list, camid=1, relabel=True)
        ir_train, _, ir_train_imgs = \
            self._process_dir(self.dataset_dir, train_ir_list, camid=2, relabel=True)
        ir_test, num_test_pids, ir_test_imgs = \
            self._process_dir(self.dataset_dir, test_rgb_list, camid=1)
        rgb_test, _, rgb_test_imgs = \
            self._process_dir(self.dataset_dir, test_ir_list, camid=2)
        num_total_pids = num_train_pids + num_test_pids
        num_total_imgs = rgb_train_imgs + ir_train_imgs + ir_test_imgs + rgb_test_imgs
        print("=> regdb loaded trial:{:2d}".format(trial))
        print("  --------------------------------")
        print("  subset       | # ids | # images")
        print("  --------------------------------")
        print("  rgb          | {:5d} | {:8d}".format(num_train_pids, rgb_train_imgs))
        print("  ir           | {:5d} | {:8d}".format(num_train_pids, ir_train_imgs))
        print("  test(ir)     | {:5d} | {:8d}".format(num_test_pids, ir_test_imgs))
        print("  test(rgb)    | {:5d} | {:8d}".format(num_test_pids, rgb_test_imgs))
        print("  total        | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  --------------------------------")
        self.train = rgb_train + ir_train
        self.rgb_test = rgb_test
        self.ir_test = ir_test
        self.testri = rgb_test + ir_test
        self.testir = ir_test + rgb_test

        self.num_train_pids = num_train_pids

    def _process_dir(self, dir_path, dir_list, camid, relabel=False):
        with open(dir_list,'rt') as f:
            data_file_list = f.read().splitlines()
            img_paths = [osp.join(dir_path,s.split(' ')[0]) for s in data_file_list]
            pids = [int(s.split(' ')[1]) for s in data_file_list]
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for idx,img_path in enumerate(img_paths):
            pid = pids[idx]
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        num_pids = len(pid_container)
        num_imgs = len(img_paths)
        return dataset, num_pids, num_imgs

class ImageDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        is_rgb = (camid==1) + (camid==2) + (camid==4) + (camid==5)
        img, _ = self.transform(img, is_rgb)
        return img, pid, camid

class ImageDatasetPath(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = Image.open(img_path).convert('RGB')
        is_rgb = (camid==1) + (camid==2) + (camid==4) + (camid==5)
        img, _ = self.transform(img, is_rgb)
        return img, pid, camid, img_path

if __name__ == '__main__':
    dataset = sysu()
    # dataset = regdb()
    exit()

