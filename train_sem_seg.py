import torch
from torch.utils.data import DataLoader
import numpy as np
from model.model import PointNet2Sem_seg, PointNet2Sem_seg_MSG
from dataset import S3DISDataset
import argparse
from torch.optim import lr_scheduler, Adam
import torch.nn.functional as F
import os

parse = argparse.ArgumentParser()
parse.add_argument('--nepoch', type=int, default=128)
parse.add_argument('--n_point', type=int, default=4096, help='Number of per sample from room')
parse.add_argument('--n_class', type=int, default=13, help='Number of class need to be seg')
parse.add_argument('--block_size', type=float, default=1.0, help='a sample group size')
parse.add_argument('--sample_rate', type=float, default=1.0)
parse.add_argument('--test_split', type=int, default=5, help='From 1-6 choice one to de test')
parse.add_argument('--batch_size', type=int, default=16)
parse.add_argument('--nworks', type=int, default=4)
parse.add_argument('--save_inter', type=int, default=20, help='Save model interval')
parse.add_argument('--output', type=str, default='/home/ouc/TXH/MyCode/PointNet++/save_model_msg', help='Path for saving model')
parse.add_argument('--root_dir', type=str, default='/home/ouc/TXH/DataSet/stanford_indoor3d/')
parse.add_argument('--msg', action='store_true', default=True, help='Control whether use msg')
# parse.add_argument('--optim', type=str, default='Adam', help='Adam or SGD')

args = parse.parse_args()

device = 'cuda:1'
train_dataset = S3DISDataset(num_points=args.n_point,
                             split='train',
                             test=args.test_split,
                             num_class=args.n_class,
                             root_dir=args.root_dir,
                             block_size=args.block_size,
                             sample_rate=args.sample_rate)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.nworks
)

test_dataset = S3DISDataset(num_points=args.n_point,
                             split='test',
                             test=args.test_split,
                             num_class=args.n_class,
                             root_dir=args.root_dir,
                             block_size=args.block_size,
                             sample_rate=args.sample_rate)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.nworks
)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)

if args.msg:
    model = PointNet2Sem_seg_MSG(args.n_class).to(device)
else:
    model = PointNet2Sem_seg(args.n_class).to(device)
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
model = model.apply(weights_init)
for epoch in range(args.nepoch):
    scheduler.step()
    for idx, data in enumerate(train_loader):
        model = model.train()
        xyz, target = data[0], data[1]
        # Error 10: need to be float32
        xyz, target = xyz.float().to(device), target.long().to(device)
        pred = model(xyz) # [B N soft_class]

        optimizer.zero_grad()
        # Error 17:RuntimeError: view size is not compatible with input tensor's size and stride
        # 此时需要加上contiguous()让他连续
        # need to do contiguous()
        # Error 18:TypeError: nll_loss(): argument 'weight' (position 3) must be Tensor, not numpy.ndarray
        # Error 19:RuntimeError: multi-target not supported
        # label必须只有一个维度
        # label must be one dim, add[:, 0]
        weight = torch.Tensor(train_dataset.label_weight).to(device)
        loss = F.nll_loss(input=pred.contiguous().view(-1, args.n_class), target=target.view(-1, 1)[:, 0], weight=weight)
        loss.backward()
        optimizer.step()


        pred_idx = torch.argmax(pred, dim=-1)

        correct = target.eq(pred_idx.data).sum()
        correct_per = correct.item() / float(target.size()[0] * target.size()[1])
        print('Epoch:%d, loss:%f, correct:%7f ' % (epoch, loss.item(), correct_per))
        # Error 20:ZeroDivisionError: integer division or modulo by zero
        # 0不能除其他
    if (epoch+1) % args.save_inter == 0:
        torch.save(model.state_dict(), os.path.join(args.output, '{}.pth'.format(epoch)))
        print('Have saved !!')

torch.save(model.state_dict(), os.path.join(args.output,'{}.pth'.format(epoch)))
print('Have saved !!')

