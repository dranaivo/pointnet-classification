import os
import numpy as np
import torch
import open3d as o3d
from PIL import Image


class PartDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, root, npoints = 2500, classification = False, class_choice = None, train = True, image=False):
        '''
        Assign the parameters: number of points, root folder, category file, image, classification.
        '''
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.image = image
        self.classification = classification
        '''
        Open the Category File and Map Folders to Categories
        '''
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        
        '''
        Select categories from the dataset. 
        ex: Call in parameters "class_choice=["Airplane"].
        '''
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        
        '''
        For every item in a specific category, assign the point, segmentation, and image.
        Basically, read the dataset and store the labels.
        '''
        self.meta = {}        
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_seg_img = os.path.join(self.root, self.cat[item], "seg_img")
            #print(dir_point, dir_seg)
            
            fns = sorted(os.listdir(dir_point))
            if train:
                fns = fns[:int(len(fns) * 0.9)]
            else:
                fns = fns[int(len(fns) * 0.9):]

            #print(os.path.basename(fns))
            for fn in fns: # FOR EVERY POINT CLOUD FILE
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg'), 
                                        os.path.join(dir_seg_img, token + '.png')))

        '''
        Create a container where you have ALL (item, points, segmentation points, segmentation image)
        '''
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2]))

        # assign the sorted class names to 0, 1, ..., #classes
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        
        self.num_seg_classes = 0
        if not self.classification: # Take the Segmentation Labels
            for i in range(len(self.datapath) // 50):
                l = len(np.unique(np.loadtxt(self.datapath[i][2]).astype(np.uint8)))
                if l > self.num_seg_classes:
                    self.num_seg_classes = l

    def __getitem__(self, index):
        '''
        This will be used to pick a specific element from the dataset.
        self.datapath is the dataset.
        Each element is under format "class, points, segmentation labels, segmentation image"
        '''
        # Get one Element
        fn = self.datapath[index]
        
        # get its Class
        cls = self.classes[fn[0]]
        
        # Read the Point Cloud
        point_set = np.asarray(o3d.io.read_point_cloud(fn[1], format='xyz').points, dtype=np.float32)
        
        # Read the Segmentation Data
        seg = np.loadtxt(fn[2]).astype(np.int64)

        #print(point_set.shape, seg.shape)
        
        # Read the Segmentation Image
        image = Image.open(fn[3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        
        #resample
        point_set = point_set[choice, :]        
        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        
        if self.classification:
            if self.image:
                return point_set, cls, image
            else:
                return point_set, cls

        else:
            if self.image:
                return point_set, seg, image
            else:
                return point_set, seg

    def __len__(self):
        return len(self.datapath)


if __name__ == "__main__":
    from utils import classes_dict, read_pointnet_colors
    import random
    import matplotlib.pyplot as plt

    # choose a GUI backend for matplotlib
    import matplotlib
    matplotlib.use('TkAgg')
    
    DATA_FOLDER = 'shapenetcore_partanno_segmentation_benchmark_v0'
    d = PartDataset(DATA_FOLDER, npoints=10000, classification=True, train=True, image=True)
    d_seg = PartDataset(DATA_FOLDER, npoints=10000, classification=False, train=True, image=False)
    print("Number of objects",len(d))
    print('----------')

    idx = random.randint(0, len(d))
    idx = 0
    _, cls, img = d[idx]
    ps, seg = d_seg[idx]

    print("Point Cloud Caracteristics:")
    print(ps.size(), ps.type(), cls.size(),cls.type())
    print('----------')
    print("Point Cloud")
    print(ps)
    print('----------')
    print("Label on Classification")
    classes_dict_list = list(classes_dict)
    print(classes_dict_list[cls.item()])

    # Visualize part labels
    plt.imshow(np.asarray(img))
    plt.show()

    # Visualize point clouds
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ps)
    pcd.colors = o3d.utility.Vector3dVector(read_pointnet_colors(seg.numpy()))

    o3d.visualization.draw_geometries([pcd])
