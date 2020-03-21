import torch
import numpy as np


def farthest_point_sample(xyz, num_sample):
    device = xyz.device
    B, N, _ = xyz.size()
    # 随机选择第一个点
    # Error 8: size(B, 1) is different to (B, )
    # (B, ) can be given
    first_point = torch.randint(0, N, (B, ), dtype=torch.long).to(device)
    farthest_index = first_point
    # 用来存放得到的点的index
    centroid_index = torch.zeros(B, num_sample, dtype=torch.long).to(device)
    # 算 没被选中的点 到 被选中的点 之间的最小距离
    dist_all = torch.ones(B, N).to(device) * 1e10
    # 使用这个就会变成每个第一维度都对应一个farthest，而不是一对一，要注意
    batch_list = torch.arange(B, dtype=torch.long).to(device)
    for i in range(num_sample):
        centroid_index[:, i] = farthest_index
        centroid_point = xyz[batch_list, farthest_index, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid_point) ** 2, dim=-1)
        # Error 8: need float 32
        mask = dist < dist_all
        dist_all[mask] = dist[mask]
        farthest_index = torch.argmax(dist_all, dim=-1)

    return centroid_index

def idx2point(xyz, idx):
    device = xyz.device
    B, N, _ = xyz.size()
    # 思路就是让batch维度中的数组维度和idx相同，达到一一对应的效果
    # 即使idx多一个维度，也没有关系，后面会做repeat
    # Error 11:'torch.Size' object does not support item assignment
    # idx.shape is torch.Size, so can't be assignment, need to be list
    batch_view = list(idx.shape)
    batch_view[1:] = [1] * (len(batch_view) - 1)
    repeat_times = list(idx.shape)
    repeat_times[0] = 1
    batch = torch.arange(B, dtype=torch.long).to(device).view(batch_view).repeat(repeat_times)
    point = xyz[batch, idx, :]

    return point

def get_distance(xyz, center_point):
    # 输出的NM和输入相反
    B, M, _ = xyz.size()
    _, N, _ = center_point.size()
    # 目标是构造一个B x N x M矩阵，存放distance，N是group的数量，M是所有点的数量
    sum_2_xyz = torch.sum(xyz ** 2, dim= -1).view(B, -1, M)
    sum_2_center = torch.sum(center_point ** 2, dim=-1).view(B, N, -1)
    sum_each = torch.bmm(center_point, xyz.permute(0, 2, 1))
    matrix_dist = sum_2_xyz + sum_2_center - 2 * sum_each

    return matrix_dist


def ball_group(xyz, center_point, radius, n_sample):
    B, N, _ = xyz.size()
    _, S, _ = center_point.shape
    matrix_dist = get_distance(xyz, center_point)
    matrix_dist_sorted, matrix_dist_sorted_idx = torch.sort(matrix_dist, dim=-1, descending=False)
    # B M S C
    group = matrix_dist_sorted[:, :, :n_sample]
    group_idx = matrix_dist_sorted_idx[:, :, :n_sample]
    # 如果范围内的点不够，就用最近的点替代，也就是排序后的第一个值
    mask = group > radius
    # 找出采样出的group中index为N的，全部设置为第一值
    # Error 12: shape not can't broadcast
    group_idx[mask] = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, n_sample)[mask]

    return group_idx


    # xyz: B N 3
def sample_and_group(xyz, feature, n_group, radius, n_sample, return_fps=False):
    '''

    :param xyz: [B, num of point, 3]
    :param feature: [B, num of point, channel]
    :param n_group:
    :param radius:
    :param n_sample:
    :param return_fps:
    :return:
    '''
    # FPS算法找形心
    fps_idx = farthest_point_sample(xyz, n_group)
    fps_point = idx2point(xyz, fps_idx)
    # 用ball找形心周围的点
    group_idx = ball_group(xyz, fps_point, radius, n_sample)
    # Error 13: careless
    group_point = idx2point(xyz, group_idx)
    # Error 14: shape not support boardcast
    group_point_norm = group_point - fps_point[:, :, None, :]
    group_point_feature = idx2point(feature, group_idx)

    if feature is not None:
        new_feature = torch.cat((group_point_norm, group_point_feature), dim=-1)
    else:
        new_feature = group_point_norm

    if return_fps:
        return fps_point, new_feature, group_point, fps_idx
    else:
        return fps_point, new_feature

def sample_and_group_all(xyz, feature):
    return True
