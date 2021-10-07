import os
import os.path
import numpy as np
import open3d
import torch
import torch.utils.data as data

class PartDataset(data.Dataset):
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        self.classification = classification

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}

        self.meta = {}
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))


        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        self.num_seg_classes = 0
        if not self.classification:
            for i in range(len(self.datapath)//50):
                l = len(np.unique(np.loadtxt(self.datapath[i][-1]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l


    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.asarray(
            open3d.io.read_point_cloud(fn[1], format='xyz').points,
            dtype=np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        point_set = point_set[choice, :]
        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    print('test')
    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', class_choice = ['Chair'])
    print(len(d))
    ps, seg = d[0]
    print(ps.size(), ps.type(), seg.size(),seg.type())

    d = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True)
    print(len(d))
    ps, cls = d[0]
    print(ps.size(), ps.type(), cls.size(),cls.type())
