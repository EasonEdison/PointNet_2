import torch
import torch.nn.functional as F
import torch.nn as nn
from model.utils import sample_and_group, sample_and_group_all, get_distance, idx2point, farthest_point_sample, ball_group

class SetAbstract(nn.Module):
    def __init__(self, n_group, n_sample, radius, in_channel, mlp_list, group_all):
        super(SetAbstract, self).__init__()
        self.n_group = n_group
        self.n_sample = n_sample
        self.radius = radius
        self.group = group_all
        self.bn_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp_list:
            self.conv_list.append(nn.Conv2d(last_channel, out_channel, 1))
            self.bn_list.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel



    # batch num channel
    def forward(self, xyz, point_feature):
        '''
        :param xyz: [B, num of point, 3]
        :param point_feature: [B, num of point, channel]
        :return: fps_point: [B, num of group, 3]
                new_feature: [B, number of group, channel]
        '''
        if self.group:
            fps_point, new_feature = sample_and_group_all(xyz, point_feature)
        else:
            fps_point, new_feature = sample_and_group(xyz, point_feature, self.n_group, self.radius, self.n_sample)

        new_feature = new_feature.permute(0, 3, 2, 1) #[B num_group num_sample C] -> [B C num_sample num_group]
        for i in range(len(self.conv_list)):
            new_feature = self.conv_list[i](new_feature)
            new_feature = self.bn_list[i](new_feature)
            new_feature = F.relu(new_feature)
        # 首尾一致
        new_feature = torch.max(new_feature, dim=2)[0] # [B C num_group]
        new_feature = new_feature.permute(0, 2, 1) # [B num_group C]

        return fps_point, new_feature

class SetAbstractMSG(nn.Module):
    def __init__(self, n_group, radius, n_sample, in_channel, mlp_list, group_all=False):
        super(SetAbstractMSG, self).__init__()
        self.n_group = n_group
        self.n_sample = n_sample
        self.radius = radius
        self.group = group_all
        self.bn_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        # Error 22:
        # 必须再循环中设置last_channel, 不然不能重置
        # must in loop to reset last_cahnnel
        # last_channel = in_channel + 3
        for i in range(len(mlp_list)):
            conv = nn.ModuleList()
            bn = nn.ModuleList()
            last_channel = in_channel + 3
            for j in range(len(mlp_list[i])):
                conv.append(nn.Conv2d(last_channel, mlp_list[i][j], 1))
                bn.append(nn.BatchNorm2d(mlp_list[i][j]))
                last_channel = mlp_list[i][j]
            self.conv_list.append(conv)
            self.bn_list.append(bn)

    def forward(self, xyz, point_feature):
        '''
        :param xyz: [B, num of point, 3]
        :param point_feature: [B, num of point, channel]
        '''
        fps_index = farthest_point_sample(xyz, self.n_group)
        fps_point = idx2point(xyz, fps_index) # [B, num_group, 3]
        new_feature_list = []
        # B is n times
        for i in range(len(self.radius)):
            ball_sample_num = self.n_sample[i]
            ball_sample_index = ball_group(xyz, fps_point, self.radius[i], ball_sample_num)
            ball_sampel_point = idx2point(xyz, ball_sample_index) # [B, group_num, sampel_num, 3]
            ball_sampel_point_norm = ball_sampel_point - fps_point[:, :, None, :]
            ball_sampel__feature = idx2point(point_feature, ball_sample_index)
            new_feature = torch.cat((ball_sampel__feature, ball_sampel_point_norm), dim=-1) # [B, group, sample, C]
            new_feature = new_feature.permute(0, 3, 1, 2)  # [B, C, group, sample]
            for j in range(len(self.conv_list[i])):
                new_feature = self.conv_list[i][j](new_feature)
                new_feature = self.bn_list[i][j](new_feature)
                new_feature = F.relu(new_feature)
            new_feature = torch.max(new_feature, dim=-1)[0]  # [B, C, group]
            new_feature = new_feature.permute(0, 2, 1) # [B, group, C]
            new_feature_list.append(new_feature)

        new_feature_list = new_feature_list
        # B x n -> B
        new_feature_list = torch.cat(new_feature_list, dim=-1)

        return fps_point, new_feature_list




class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp_list):
        super(FeaturePropagation, self).__init__()
        self.in_channel = in_channel
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for i in range(len(mlp_list)):
            self.conv_list.append(nn.Conv1d(in_channel, mlp_list[i], 1))
            self.bn_list.append(nn.BatchNorm1d(mlp_list[i]))
            in_channel = mlp_list[i]

    def forward(self, inter_to_xyz, inter_from_xyz, inter_to_feature, inter_from_feature):
        '''
        :param inter_to_xyz: 被插值的坐标，点多的那个
        :param inter_from_xyz: 插值的坐标，点少的那个
        :param inter_to_feature: 被插值的特征，点多的那个
        :param inter_from_feature: 插值的特征，点少的那个
        :return:
        '''
        B, N, _ = inter_to_xyz.size()
        _, S, _ = inter_from_xyz.size()

        if S == 1:
            inter_get = inter_from_feature.repeat(1, N, 1)
        else:
            matrix_dist = get_distance(inter_from_xyz, inter_to_xyz) # [B, N, S]
            dist_sort, dist_sort_idx = torch.sort(matrix_dist, dim=-1)
            dist_sort = dist_sort[:, :, :3] # [B, N, 3]
            dist_sort_idx = dist_sort_idx[:, :, :3] # [B, N, 3]
            weight = 1.0 / (dist_sort + 1e-8)
            # Error 15: use keepdim to support boardcast
            weight_norm = weight / torch.sum(weight, dim=-1, keepdim=True) # [B, N, 3]
            feature_get = idx2point(inter_from_feature, dist_sort_idx) # [B, N, 3, C]
            inter_get = torch.sum(feature_get * weight_norm[:, :, :, None], dim=2) # [B, N, C]

        if inter_to_feature is not None:
            new_feature = torch.cat((inter_to_feature, inter_get), dim=-1)
        else:
            new_feature = inter_get

        new_feature = new_feature.permute(0, 2, 1) # [B, N, C] -> [B, C, N]


        for i in range(len(self.conv_list)):
            new_feature = self.conv_list[i](new_feature)
            new_feature = self.bn_list[i](new_feature)
            # Error 21: after relu , be nan
            # in weight / same thing add 1e-8
            new_feature = F.relu(new_feature)
        new_feature = new_feature.permute(0, 2, 1) # [B, C, N] -> [B, N, C]

        return new_feature

class PointNet2Sem_seg(nn.Module):
    def __init__(self, num_class):
        super(PointNet2Sem_seg, self).__init__()
        self.abstract_1 = SetAbstract(1024, 32, 0.1, 9+3, [32, 32, 64], False)
        self.abstract_2 = SetAbstract(256, 32, 0.2, 64+3, [64, 64, 128], False)
        self.abstract_3 = SetAbstract(64, 32, 0.4, 128+3, [128, 128, 256], False)
        self.abstract_4 = SetAbstract(16, 32, 0.8, 256+3, [256, 256, 512], False)
        self.propaga_1 = FeaturePropagation(768, [256, 256])
        self.propaga_2 = FeaturePropagation(384, [256, 256])
        self.propaga_3 = FeaturePropagation(320, [256, 128])
        self.propaga_4 = FeaturePropagation(128, [256, 128, 128])
        self.conv_1 = nn.Conv1d(128, 128,1)
        self.bn_1 = nn.BatchNorm1d(128)
        self.dropput = nn.Dropout(p=0.5)
        self.conv_2 = nn.Conv1d(128, num_class, 1)

    def forward(self, x):
        # B N C
        xyz = x[:, :, :3]
        l_point_1, l_feature_1 = self.abstract_1(xyz, x)
        l_point_2, l_feature_2 = self.abstract_2(l_point_1, l_feature_1)
        l_point_3, l_feature_3 = self.abstract_3(l_point_2, l_feature_2)
        l_point_4, l_feature_4 = self.abstract_4(l_point_3, l_feature_3)

        inter_feat_1 = self.propaga_1(l_point_3, l_point_4, l_feature_3, l_feature_4)
        inter_feat_2 = self.propaga_2(l_point_2, l_point_3, l_feature_2, inter_feat_1)
        inter_feat_3 = self.propaga_3(l_point_1, l_point_2, l_feature_1, inter_feat_2)
        inter_feat_4 = self.propaga_4(xyz, l_point_1, None, inter_feat_3) # [B N C]

        new_feature = inter_feat_4.permute(0, 2, 1)
        new_feature = self.conv_1(new_feature)
        new_feature = self.bn_1(new_feature)
        # Error 21: forget relu
        new_feature = F.relu(new_feature)
        new_feature = self.dropput(new_feature)
        # Error 16: Conv1d
        new_feature = self.conv_2(new_feature) # [B num_class N]
        result_seg = F.log_softmax(new_feature, dim=1)
        result_seg = result_seg.permute(0, 2, 1) # [B N soft_class]

        return result_seg

class PointNet2Sem_seg_MSG(nn.Module):
    def __init__(self, num_class):
        super(PointNet2Sem_seg_MSG, self).__init__()
        self.abstract_1 = SetAbstractMSG(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.abstract_2 = SetAbstractMSG(256, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.abstract_3 = SetAbstractMSG(64, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.abstract_4 = SetAbstractMSG(16, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]])
        self.propaga_1 = FeaturePropagation(512+512+256+256, [256, 256])
        self.propaga_2 = FeaturePropagation(128+128+256, [256, 256])
        self.propaga_3 = FeaturePropagation(32+64+256, [256, 128])
        self.propaga_4 = FeaturePropagation(128, [128, 128, 128])


        self.conv_1 = nn.Conv1d(128, 128, 1)
        self.bn_1 = nn.BatchNorm1d(128)
        self.dropput = nn.Dropout(p=0.5)
        self.conv_2 = nn.Conv1d(128, num_class, 1)



    def forward(self, x):
        # B N C
        xyz = x[:, :, :3]
        l_point_1, l_feature_1 = self.abstract_1(xyz, x)
        l_point_2, l_feature_2 = self.abstract_2(l_point_1, l_feature_1)
        l_point_3, l_feature_3 = self.abstract_3(l_point_2, l_feature_2)
        l_point_4, l_feature_4 = self.abstract_4(l_point_3, l_feature_3)

        inter_feat_1 = self.propaga_1(l_point_3, l_point_4, l_feature_3, l_feature_4)
        inter_feat_2 = self.propaga_2(l_point_2, l_point_3, l_feature_2, inter_feat_1)
        inter_feat_3 = self.propaga_3(l_point_1, l_point_2, l_feature_1, inter_feat_2)
        inter_feat_4 = self.propaga_4(xyz, l_point_1, None, inter_feat_3) # [B N C]

        new_feature = inter_feat_4.permute(0, 2, 1)
        new_feature = self.conv_1(new_feature)
        new_feature = self.bn_1(new_feature)
        # Error 21: forget relu
        new_feature = F.relu(new_feature)
        new_feature = self.dropput(new_feature)
        # Error 16: Conv1d
        new_feature = self.conv_2(new_feature) # [B num_class N]
        result_seg = F.log_softmax(new_feature, dim=1)
        result_seg = result_seg.permute(0, 2, 1) # [B N soft_class]

        return result_seg
