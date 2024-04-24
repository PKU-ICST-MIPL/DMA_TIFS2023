import time
import argparse
from utils import *
from loss import hcc
import torch.nn as nn
from model import resnet
import torch.optim as optim
from torch.nn import functional as F
from eval import eval, extract_feature
from torch.utils.data import DataLoader
from datamanager import ImageDataset, sysu, regdb
from sample import sysuSampler, sysugallerySampler, regdbSampler

parser = argparse.ArgumentParser(description='Reid')
parser.add_argument('--dataset','-d', default='sysu',help='sysu, or regdb')
parser.add_argument('--log_path', default='./log/', help='log save path')#
parser.add_argument('--model_path', default='./save_model/',help='model save path')#
parser.add_argument('-optim', default='sgd', help='sgd or adam')
parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
parser.add_argument('-dlr', default=10, type=float, help='discriminative learning rates')
parser.add_argument('--max_epoch', '-maxe', default=120, type=int, help='maximum epochs')
parser.add_argument('--warmup_epoch', '-ware', default=10, type=int, help='>=0')
parser.add_argument('--scheduler_epoch', '-sche', default=[60, 100], type=int, nargs='*',
                    help='lr scheduler by Step')
parser.add_argument('--test_epoch', default=10, type=int, help='test model every 10 epochs, not equal to real test')
parser.add_argument('--save_epoch', default=120, type=int, help='save model every 20 epochs')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--img_h', '-ih', default=384, type=int, help='img height')
parser.add_argument('--img_w', '-iw', default=192, type=int, help='img width')
parser.add_argument('--num_id', '-p', default=8, type=int, help='num of identity per batch')
parser.add_argument('--num_pos', '-k', default=4, type=int, help='num of pos per identity')
parser.add_argument('--test_batch', default=256, type=int, help='testing batch size')
parser.add_argument('-test', action='store_true', help='no logs and models are saved')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--trial', default=0, type=int, help='trial')
# parser.add_argument('-gpu', default='7', help='gpu device ids for CUDA_VISIBLE_DEVICES')#
parser.add_argument('--version', '-v', default='1', type=int, help='version')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu()
set_seed(args.seed)
if args.dataset == 'sysu':
    pre_process_sysu()
time_offset = 12*3600
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+time_offset))
# if not os.path.isdir(args.log_path):
#     os.makedirs(args.log_path)
# if not os.path.isdir(args.model_path):
#     os.makedirs(args.model_path)
suffix = f'{args.dataset}'
# while os.path.exists(args.log_path + suffix + '_v' + str(args.version) + '.txt'):
#     if os.path.isfile(args.model_path + suffix + '_v' + str(args.version) + '/complete.txt'):
#         args.version += 1
#     else:
#         os.remove(args.log_path + suffix + '_v' + str(args.version) + '.txt')
suffix += '_v' + str(args.version)
checkpoint_path = args.model_path + suffix + '/'

# if args.test == False:
#     if not os.path.isdir(checkpoint_path):
#         os.makedirs(checkpoint_path)
#     sys.stdout = Logger(args.log_path + suffix + '.txt')

print(suffix)
if args.dataset == 'sysu':
    dataset = sysu()
elif args.dataset == 'regdb':
    dataset = regdb(args.trial)

# exit(0)
criterion_id = nn.CrossEntropyLoss()
criterion_id.to('cuda')
criterion_hcc = hcc(margin_euc=0.6, margin_kl=6)
criterion_hcc.to('cuda')
net = resnet(class_num=dataset.num_train_pids)
net.to('cuda')

ignored_params = list(map(id, net.ignored_params.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
optimizer = optim.SGD([
    {'params': base_params, 'lr': args.lr},
    {'params': net.ignored_params.parameters(), 'lr': args.dlr * args.lr},],
    weight_decay=5e-4, momentum=0.9, nesterov=True)
transform_train, transform_test = get_transform(args)

if args.dataset == 'sysu':
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=sysuSampler(dataset.train, args.num_id, args.num_pos),######################################################################
        batch_size=args.num_id * args.num_pos * 2,
        num_workers=args.workers, pin_memory=True, drop_last=True,)
elif args.dataset == 'regdb':
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=regdbSampler(dataset.train, args.num_id, args.num_pos),######################################################################
        batch_size=args.num_id * args.num_pos * 2,
        num_workers=args.workers, pin_memory=True, drop_last=True,)

def test(net, queryloader, galleryloader):
    q_pool, q_feat, q_pids, q_camids = extract_feature(net, queryloader)
    g_pool, g_feat, g_pids, g_camids = extract_feature(net, galleryloader)
    q = F.normalize(q_feat, p=2, dim=1)
    g = F.normalize(g_feat, p=2, dim=1)
    distmat = -torch.mm(q, g.t()).numpy()
    cmc1, mAP1, mINP1 = eval(distmat, q_pids, g_pids, q_camids, g_camids)
    return cmc1, mAP1, mINP1

print('==> Start Testing...', suffix, 'gpu:', os.environ['CUDA_VISIBLE_DEVICES'])

pre_trained_model = torch.load(args.model_path)
net.load_state_dict(pre_trained_model['net'])


print(f'==> Start Testing')
start = time.time()
if args.dataset == 'sysu':##################################################################################################
    t = 10
    for trial in range(t):
        queryloader = DataLoader(
            ImageDataset(dataset.query, transform=transform_test), batch_size=args.test_batch,
            shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,)
        galleryloader = DataLoader(
            ImageDataset(dataset.gallery, transform=transform_test),
            sampler=sysugallerySampler(dataset.gallery, trial, shot=1, mode='all'),
            batch_size=args.test_batch, num_workers=args.workers,
            pin_memory=True, drop_last=False,)
        galleryloader_10 = DataLoader(
            ImageDataset(dataset.gallery, transform=transform_test),
            sampler=sysugallerySampler(dataset.gallery, trial, shot=10, mode='all'),
            batch_size=args.test_batch, num_workers=args.workers,
            pin_memory=True, drop_last=False,)
        galleryloader_indoor = DataLoader(
            ImageDataset(dataset.gallery, transform=transform_test),
            sampler=sysugallerySampler(dataset.gallery, trial, shot=1, mode='indoor'),
            batch_size=args.test_batch, num_workers=args.workers,
            pin_memory=True, drop_last=False,)
        galleryloader_10_indoor = DataLoader(
            ImageDataset(dataset.gallery, transform=transform_test),
            sampler=sysugallerySampler(dataset.gallery, trial, shot=10, mode='indoor'),
            batch_size=args.test_batch, num_workers=args.workers,
            pin_memory=True, drop_last=False,)
        cmc_t, mAP_t, mINP_t = test(net, queryloader, galleryloader)
        cmc_t_10, mAP_t_10, mINP_t_10 = test(net, queryloader, galleryloader_10)
        cmc_t_indoor, mAP_t_indoor, mINP_t_indoor = test(net, queryloader, galleryloader_indoor)
        cmc_t_10_indoor, mAP_t_10_indoor, mINP_t_10_indoor = test(net, queryloader, galleryloader_10_indoor)
        if trial == 0:
            cmc, mAP, mINP = cmc_t/t, mAP_t/t, mINP_t/t
            cmc_10, mAP_10, mINP_10 = cmc_t_10/t, mAP_t_10/t, mINP_t_10/t
            cmc_indoor, mAP_indoor, mINP_indoor = cmc_t_indoor/t, mAP_t_indoor/t, mINP_t_indoor/t
            cmc_10_indoor, mAP_10_indoor, mINP_10_indoor = cmc_t_10_indoor/t, mAP_t_10_indoor/t, mINP_t_10_indoor/t
        else:
            cmc, mAP, mINP = cmc + cmc_t/t, mAP + mAP_t/t, mINP + mINP_t/t
            cmc_10, mAP_10, mINP_10 = cmc_10 + cmc_t_10/t, mAP_10 + mAP_t_10/t, mINP_10 + mINP_t_10/t
            cmc_indoor, mAP_indoor, mINP_indoor = cmc_indoor + cmc_t_indoor/t, mAP_indoor + mAP_t_indoor/t, mINP_indoor + mINP_t_indoor/t
            cmc_10_indoor, mAP_10_indoor, mINP_10_indoor = cmc_10_indoor + cmc_t_10_indoor/t, mAP_10_indoor + mAP_t_10_indoor/t, mINP_10_indoor + mINP_t_10_indoor/t

    print (f'All search & Single-Shot:')
    print(f'Rank-1: {cmc[0]:.2%} | Rank-10: {cmc[9]:.2%} | Rank-20: {cmc[19]:.2%} | mAP: {mAP:.2%}')
    print (f'All search & multi-Shot:')
    print(f'Rank-1: {cmc_10[0]:.2%} | Rank-10: {cmc_10[9]:.2%} | Rank-20: {cmc_10[19]:.2%} | mAP: {mAP_10:.2%}')
    print (f'Indoor search & Single-Shot:')
    print(f'Rank-1: {cmc_indoor[0]:.2%} | Rank-10: {cmc_indoor[9]:.2%} | Rank-20: {cmc_indoor[19]:.2%} | mAP: {mAP_indoor:.2%}')
    print (f'Indoor search & multi-Shot:')
    print(f'Rank-1: {cmc_10_indoor[0]:.2%} | Rank-10: {cmc_10_indoor[9]:.2%} | Rank-20: {cmc_10_indoor[19]:.2%} | mAP: {mAP_10_indoor:.2%}')

    

elif args.dataset == 'regdb':###############################################################################################
    queryloader = DataLoader(
        ImageDataset(dataset.rgb_test, transform=transform_test), batch_size=args.test_batch,
        shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,)
    galleryloader = DataLoader(
        ImageDataset(dataset.ir_test, transform=transform_test), batch_size=args.test_batch,
        shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,)
    
    print (f'Visible to Infrared:')
    cmc, mAP, mINP = test(net, galleryloader, queryloader)
    print(f'Rank-1: {cmc[0]:.2%} | Rank-5: {cmc[4]:.2%} | Rank-10: {cmc[9]:.2%} | '
        f'Rank-20: {cmc[19]:.2%} | mAP: {mAP:.2%} | mINP: {mINP:.2%} '
        f'Evaluation Time: {time.time() - start:.3f}')
    tmp_best_r1 = cmc[0]

    print (f'Infrared to Visible:')
    cmc, mAP, mINP = test(net, queryloader, galleryloader)
    print(f'Rank-1: {cmc[0]:.2%} | Rank-5: {cmc[4]:.2%} | Rank-10: {cmc[9]:.2%} | '
        f'Rank-20: {cmc[19]:.2%} | mAP: {mAP:.2%} | mINP: {mINP:.2%} '
        f'Evaluation Time: {time.time() - start:.3f}')
        # torch.save(state, checkpoint_path + 'epoch_{}.t'.format(epoch))

if __name__ == '__main__':
    pass
