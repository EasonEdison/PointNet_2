import numpy as np
import os
from torch.utils.data import Dataset


class S3DISDataset(Dataset):
    def __init__(self, num_points=4096,
                 split='train',
                 root_dir=None,
                 test=6,
                 block_size=1.0,
                 sample_rate=1.0,
                 transform_control=None,
                 num_class=13
                 ):
        super(S3DISDataset, self).__init__()
        self.rooms_point = []
        self.rooms_label = []
        self.rooms_min_coor = []
        self.rooms_max_coor = []
        self.rooms_index = []
        self.block_size = block_size
        self.num_points = num_points
        self.transform_control = transform_control
        num_points_all = []

        rooms_dir = os.listdir(root_dir)
        # get file name
        if split == 'train':
            rooms = [room for room in rooms_dir if 'Area_{}'.format(test) not in room]
        else:
            rooms = [room for room in rooms_dir if 'Area_{}'.format(test) in room]
        # total 13 classes
        label_num = np.zeros(num_class)

        # 根据room来划分
        for i in range(len(rooms)):
            points = np.load(os.path.join(root_dir, rooms[i]))
            self.rooms_point.append(points[:, :6])
            self.rooms_label.append(points[:, -1])
            self.rooms_min_coor.append(np.amin(points[:, :3], axis=0))
            self.rooms_max_coor.append(np.amax(points[:, :3], axis=0))
            # Error 1: second parameter : 'range(14)' = 'bins=13'
            number_labels, _ = np.histogram(points[:, -1], bins=num_class)
            label_num += number_labels.astype(np.float32)
            # 每次只计算room中点的总数
            num_points_all.append(points.shape[0])

        # 计算label权重
        self.label_weight = label_num / np.sum(label_num)
        # Error 2:TypeError: return arrays must be of ArrayType
        # self.label_weight = np.sqrt((np.amax(self.label_weight)/ self.label_weight), 1/3)
        # careless
        self.label_weight = np.power((np.amax(self.label_weight)/ self.label_weight), 1/3)

        # 分配room比例
        sample_per_room = num_points_all / np.sum(num_points_all).astype(np.float)
        # 共能采样多少组点
        number_group_per_room_sample = (np.sum(num_points_all) * sample_rate) / num_points
        # 每个room能分到多少组
        rooms_index = []
        for i in range(len(num_points_all)):
            rooms_index.extend([i] * int(round(sample_per_room[i] * number_group_per_room_sample)))
        self.rooms_index = np.array(rooms_index)

        print('Load dataset over! Total rooms: {} , Total points: {}'.format(len(num_points_all), np.sum(num_points_all)))


    def __getitem__(self, index):
        room_index = self.rooms_index[index]
        points = self.rooms_point[room_index]
        labels = self.rooms_label[room_index]
        num_points = points.shape[0]
        # 随机一个中心点，以中心点为中心采样
        while True:

            center = points[np.random.choice(num_points), :3]
            # Error 3:careless
            block_points_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_points_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            # 不知这样可不可行，能广播吗？
            # 不行
            # Error 4:ufunc 'bitwise_and' not supported for the input types
            # & 需要两边都有括号
            # [False,True] & [True,False] = True
            # (n,) need to [0]
            select_points_block_index = np.where((points[:, 0] <= block_points_max[0]) & (points[:, 0] > block_points_min[0])
                                                 & (points[:, 1] <= block_points_max[1]) & (points[:, 1] >= block_points_min[1]))[0]
            # why 1024?
            if len(select_points_block_index) > 1024:
                break

        # 在选定的block中采样
        if len(select_points_block_index) > self.num_points:
            select_points_index = np.random.choice(select_points_block_index, self.num_points, replace=False)
        else:
            select_points_index = np.random.choice(select_points_block_index, self.num_points, replace=True)

        select_points = points[select_points_index, :]
        select_labels = labels[select_points_index]

        feature_points = np.zeros([select_points.shape[0], 9],)
        # 广播否 yes
        # Error 7: self.rooms_max_coor is for room, so need room_index
        feature_points[:, 6:9] = select_points[:, :3] / self.rooms_max_coor[room_index]
        feature_points[:, 3:6] = select_points[:, -3:] / 255.0
        feature_points[:, :3] = select_points[:, :3] - center

        if self.transform_control:
            feature_points, select_labels = self.transform(feature_points, select_labels)

        return feature_points, select_labels



    def transform(self, points, labels):
        return points, labels


    def __len__(self):
        return len(self.rooms_index)
