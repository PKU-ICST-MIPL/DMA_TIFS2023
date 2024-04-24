import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler
import random
import copy

class sysuSampler(Sampler):
    def __init__(self, data_source, num_p, num_k):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_k = num_k
        self.num_p = num_p
        self.index_rgb = defaultdict(list) # {pid1:[index11,index12,index13],pid2:[]}
        self.index_ir = defaultdict(list)
        self.num_rgb = 0
        self.num_ir = 0
        for index, (_, pid, camid) in enumerate(self.data_source):
            if camid in [1,2,4,5]:
                self.index_rgb[pid].append(index)
                self.num_rgb += 1
            elif camid in [3,6]:
                self.index_ir[pid].append(index)
                self.num_ir += 1
        self.pids = list(self.index_rgb.keys())
        self.length = min(self.num_rgb, self.num_ir) * 2

    def __iter__(self):
        batch_idxs_rgb = defaultdict(list)
        batch_idxs_ir = defaultdict(list)
        for pid in self.pids:
            idxs_rgb = copy.deepcopy(self.index_rgb[pid])
            idxs_ir = copy.deepcopy(self.index_ir[pid])
            if len(idxs_rgb) < self.num_k:
                idxs_rgb = np.random.choice(idxs_rgb, size=self.num_k, replace=True)
            if len(idxs_ir) < self.num_k:
                idxs_ir = np.random.choice(idxs_ir, size=self.num_k, replace=True)
            np.random.shuffle(idxs_rgb)
            np.random.shuffle(idxs_ir)
            batch_idxs = []
            for idx in idxs_rgb:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_k:
                    batch_idxs_rgb[pid].append(batch_idxs)
                    batch_idxs = []
            batch_idxs = []
            for idx in idxs_ir:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_k:
                    batch_idxs_ir[pid].append(batch_idxs)
                    batch_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_p:
            selected_pids = np.random.choice(avai_pids, self.num_p, replace=False)

            for pid in selected_pids:
                batch_idxs = batch_idxs_rgb[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_rgb[pid]) == 0:
                    avai_pids.remove(pid)
            for pid in selected_pids:
                batch_idxs = batch_idxs_ir[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_ir[pid]) == 0 and pid in avai_pids:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

class sysugallerySampler(Sampler):
    def __init__(self, data_source, trial = -1, shot = 1, mode='all'):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_dic = defaultdict(list)
        self.ret = []
        for index, (_, pid, camid) in enumerate(self.data_source):
            if mode == 'indoor' and camid in [4, 5]:
                continue
            self.index_dic[pid,camid].append(index)
            self.ret.append(index)
        self.pids = list(self.index_dic.keys())
        self.trial = trial
        self.shot = shot
        if self.trial >= 0: self.N = len(self.pids)
        else: self.N = len(self.data_source)

    def __iter__(self):
        if self.trial >= 0:
            random.seed(self.trial)#测试部分用random训练部分用np.random
            ret = []
            for pid in self.pids:
                ret.extend(random.sample(self.index_dic[pid], k=self.shot))
            #np.save('index.npy',np.array(ret))#用于可视化
        else:
            ret = self.ret
        return iter(ret)
    def __len__(self):
        return self.N

class regdbSampler(Sampler):
    def __init__(self, data_source, num_p, num_k):
        super().__init__(data_source)
        self.data_source = data_source
        self.num_k = num_k
        self.num_p = num_p
        self.index_rgb = defaultdict(list) # {pid1:[index11,index12,index13],pid2:[]}
        self.index_ir = defaultdict(list)
        self.num_rgb = 0
        self.num_ir = 0
        for index, (_, pid, camid) in enumerate(self.data_source):
            if camid == 1:
                self.index_rgb[pid].append(index)
                self.num_rgb += 1
            elif camid == 2:
                self.index_ir[pid].append(index)
                self.num_ir += 1
        self.pids = list(self.index_rgb.keys())
        self.length = min(self.num_rgb, self.num_ir)*2

    def __iter__(self):
        batch_idxs_rgb = defaultdict(list)
        batch_idxs_ir = defaultdict(list)
        for pid in self.pids:
            idxs_rgb = copy.deepcopy(self.index_rgb[pid])
            idxs_ir = copy.deepcopy(self.index_ir[pid])
            np.random.shuffle(idxs_rgb)
            np.random.shuffle(idxs_ir)
            batch_idxs = []
            for idx in idxs_rgb:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_k:
                    batch_idxs_rgb[pid].append(batch_idxs)
                    batch_idxs = []
            batch_idxs = []
            for idx in idxs_ir:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_k:
                    batch_idxs_ir[pid].append(batch_idxs)
                    batch_idxs = []
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_p:
            selected_pids = np.random.choice(avai_pids, self.num_p, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_rgb[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_rgb[pid]) == 0:
                    avai_pids.remove(pid)
            for pid in selected_pids:
                batch_idxs = batch_idxs_ir[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_ir[pid]) == 0 and pid in avai_pids:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length

if __name__ == '__main__':
    pass



