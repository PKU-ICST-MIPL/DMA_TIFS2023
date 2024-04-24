import time
import argparse
from utils import *
from loss import hcc, ptcc
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
parser.add_argument('--max_epoch', '-maxe', default=160, type=int, help='maximum epochs')
parser.add_argument('--warmup_epoch', '-ware', default=10, type=int, help='>=0')
parser.add_argument('--scheduler_epoch', '-sche', default=[60, 100], type=int, nargs='*',
                    help='lr scheduler by Step')
parser.add_argument('--test_epoch', default=10, type=int, help='test model every 10 epochs, not equal to real test')
parser.add_argument('--save_epoch', default=20, type=int, help='save model every 20 epochs')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
parser.add_argument('--img_h', '-ih', default=384, type=int, help='img height')
parser.add_argument('--img_w', '-iw', default=192, type=int, help='img width')
parser.add_argument('--part_num', '-pn', default=6, type=int, help='img part')
parser.add_argument('--num_id', '-p', default=8, type=int, help='num of identity per batch')
parser.add_argument('--num_pos', '-k', default=4, type=int, help='num of pos per identity')
parser.add_argument('--test_batch', default=256, type=int, help='testing batch size')
parser.add_argument('-test', action='store_true', help='no logs and models are saved')
parser.add_argument('--seed', default=0, type=int, help='random seed')
# parser.add_argument('-gpu', default='0', help='gpu device ids for CUDA_VISIBLE_DEVICES')#
parser.add_argument('--version', '-v', default='1', type=int, help='version')
args = parser.parse_args()
# if args.gpu != '-1':
os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu()
set_seed(args.seed)
if args.dataset == 'sysu':
    pre_process_sysu()
time_offset = 12*3600
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+time_offset))
if not os.path.isdir(args.log_path):
    os.makedirs(args.log_path)
if not os.path.isdir(args.model_path):
    os.makedirs(args.model_path)
suffix = f'{args.dataset}'
while os.path.exists(args.log_path + suffix + '_v' + str(args.version) + '.txt'):
    if os.path.isfile(args.model_path + suffix + '_v' + str(args.version) + '/complete.txt'):
        args.version += 1
    else:
        os.remove(args.log_path + suffix + '_v' + str(args.version) + '.txt')
suffix += '_v' + str(args.version)
checkpoint_path = args.model_path + suffix + '/'

if args.test == False:
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    sys.stdout = Logger(args.log_path + suffix + '.txt')

print(suffix)
if args.dataset == 'sysu':
    dataset = sysu()
elif args.dataset == 'regdb':
    dataset = regdb()

# exit(0)
criterion_id = nn.CrossEntropyLoss()
criterion_id.to('cuda')
criterion_hcc = hcc(margin_euc=0.6, margin_kl=6)
criterion_hcc.to('cuda')
criterion_pp = ptcc(margin_euc=0.3, margin_kl=6)
criterion_pp.to('cuda')
# criterion_cos = nn.CosineEmbeddingLoss(margin=-1.)
# criterion_cos.to('cuda')
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
        sampler=sysuSampler(dataset.train, args.num_id, args.num_pos),
        batch_size=args.num_id * args.num_pos * 2,
        num_workers=args.workers, pin_memory=True, drop_last=True,)
elif args.dataset == 'regdb':
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=regdbSampler(dataset.train, args.num_id, args.num_pos),
        batch_size=args.num_id * args.num_pos * 2,
        num_workers=args.workers, pin_memory=True, drop_last=True,)

def train(epoch):
    current_lr = adjust_learning_rate(args, optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    euc_loss = AverageMeter()
    kl_loss = AverageMeter()
    # rank_loss = AverageMeter()
    # cos_loss = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct_id, total = 0, 0
    net.train()
    end = time.time()
    for batch_idx, (imgs, pids, camids) in enumerate(trainloader):
        optimizer.zero_grad()
        ###############################################################################################
        # imgs_ = torch.flip(imgs, dims=[0])# b, 3, h, w
        # imgs_l = imgs.view(imgs.size(0), imgs.size(1), 1, args.part_num, imgs.size(2)//args.part_num, imgs.size(3))# b, 3, 1, pn, h/pn, w
        # imgs_l_= imgs_.view(imgs_.size(0), imgs_.size(1), 1, args.part_num, imgs_.size(2)//args.part_num, imgs_.size(3))# b, 3, 1, pn, h/pn, w
        # imgs_list = torch.cat([imgs_l, imgs_l_], dim=2)# b, 3, 2, pn, h/pn, w
        # imgs_list = imgs_list.view(imgs_list.size(0), imgs_list.size(1), imgs_list.size(2)*imgs_list.size(3), imgs_list.size(4), imgs_list.size(5))# b, 3, 2*pn, h/pn, w

        # rdn_list = [i for i in range(args.part_num)]
        # mix_imgs = []
        # for i in range(imgs.size(0)):
        #     random.shuffle(rdn_list)
        #     select = torch.arange(args.part_num)
        #     select[rdn_list[args.part_num//2:]] += args.part_num # 以多大的概率选另一个id的part
        #     mix_imgs.append(imgs_list[i, :, select.type(torch.long), :, :].view(1, imgs.size(1), imgs.size(2), imgs.size(3)))
        # mix_imgs = torch.cat(mix_imgs, 0)
        # mix_imgs = mix_imgs.cuda()
        ###############################################################################################
        imgs = imgs.cuda()
        pids = pids.cuda()
        data_time.update(time.time() - end)
        end = time.time()

        out = net(imgs, camids, pids)
        # out = net(torch.cat([imgs, mix_imgs], 0), camids, pids)####################################
        pool, y, pp = out[0], out[1], out[2]
        # pool, pool_mix = pool[:len(pool)//2], pool[len(pool)//2:]############################
        # y, y_mix = y[:len(y)//2], y[len(y)//2:]############################

        loss_id = criterion_id(y, pids)
        loss_hcc_euc = criterion_hcc(pool, pids, 'euc')
        loss_hcc_kl = criterion_hcc(y, pids, 'kl')
        loss_pp_euc = 0
        for i in range(pp.size(1)):
            loss_pp_euc += criterion_pp(pp[:,i], pids, 'euc') / pp.size(1)
        # print (loss_hcc_euc, loss_pp_euc)
        # exit(0)
        # ps1, ps2 = feat.size(0), feat.size(1)
        # pids_ = pids.view(-1,1)
        # feat_order = torch.cat([feat[:ps1//2:],feat[ps1//2::]], -1).view(ps1, ps2)
        # pids_order = torch.cat([pids_[:ps1//2:],pids_[ps1//2::]], -1).view(ps1)
        # loss_rank = criterion_rank(feat_order, pids_order)

        loss = loss_id + loss_hcc_euc + loss_hcc_kl + 0.05 * loss_pp_euc

        _, predicted = y.max(1)
        correct_id += predicted.eq(pids).sum().item()

        loss.backward()
        optimizer.step()
        ###############################################################################################
        # optimizer.zero_grad()
        # imgs_ = torch.flip(imgs, dims=[0])# b, 3, h, w
        # imgs_l = imgs.view(imgs.size(0), imgs.size(1), 1, args.part_num, imgs.size(2)//args.part_num, imgs.size(3))# b, 3, 1, pn, h/pn, w
        # imgs_l_= imgs_.view(imgs_.size(0), imgs_.size(1), 1, args.part_num, imgs_.size(2)//args.part_num, imgs_.size(3))# b, 3, 1, pn, h/pn, w
        # imgs_list = torch.cat([imgs_l, imgs_l_], dim=2)# b, 3, 2, pn, h/pn, w
        # imgs_list = imgs_list.view(imgs_list.size(0), imgs_list.size(1), imgs_list.size(2)*imgs_list.size(3), imgs_list.size(4), imgs_list.size(5))# b, 3, 2*pn, h/pn, w

        # rdn_list = [i for i in range(args.part_num)]
        # mix_imgs = []
        # for i in range(imgs.size(0)):
        #     random.shuffle(rdn_list)
        #     select = torch.arange(args.part_num)
        #     select[rdn_list[args.part_num//2:]] += args.part_num # 以多大的概率选另一个id的part
        #     mix_imgs.append(imgs_list[i, :, select.type(torch.long), :, :].view(1, imgs.size(1), imgs.size(2), imgs.size(3)))
        # mix_imgs = torch.cat(mix_imgs, 0)
        # mix_imgs = mix_imgs.cuda()
        
        # out_mix = net(mix_imgs, camids, pids)
        # v0, _ = out_mix[0], out_mix[1]
        # v1, v2 = pool.detach(), torch.flip(pool, dims=[0]).detach()
        # target = torch.ones(v0.size(0))*-1.
        # loss_cos = criterion_cos(v1-v0, v2-v0, target.cuda())/2.0
        # loss_cos.backward()
        # optimizer.step()
        ###############################################################################################

        id_loss.update(loss_id.item(), pids.size(0))
        euc_loss.update(loss_hcc_euc.item(), pids.size(0))
        kl_loss.update(loss_hcc_kl.item(), pids.size(0))
        # rank_loss.update(loss_rank.item(), pids.size(0))
        # cos_loss.update(loss_cos.item(), pids.size(0))
        train_loss.update(loss.item(), pids.size(0))
        total += pids.size(0)
        batch_time.update(time.time() - end)
        end = time.time()
        if (batch_idx + 1) % (len(trainloader) // 10) == 0:
            print(f'E[{epoch:02d}][{batch_idx + 1:3d}/{len(trainloader)}] '
                  f'L: {train_loss.val:.3f} ({train_loss.avg:.3f}) '
                  f'L_id: {id_loss.val:.3f} ({id_loss.avg:.3f}) '
                  f'L_euc: {euc_loss.val:.3f} ({euc_loss.avg:.3f}) '
                  f'L_kl: {kl_loss.val:.3f} ({kl_loss.avg:.3f}) '
                #   f'L_rank: {rank_loss.val:.3f} ({rank_loss.avg:.3f}) '
                #   f'L_cos: {cos_loss.val:.3f} ({cos_loss.avg:.3f}) '
                  f'Acc: {100. * correct_id / total:.2f} ')
                #   f'grad_norm: {grad.val:.3f} ({grad.avg:.3f}) ')
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)#torch.cuda.memory_reserved()
    print(f'Epoch[{epoch:02d}][{batch_idx + 1:3d}/{len(trainloader)}] '
          f'LR: {current_lr:.4f} '
          f'Loss: {train_loss.avg:.3f} '
          f'Acc: {100. * correct_id / total:.2f} '
        #   f'grad_norm: {grad.avg:.3f} '
          f'DataTime: {data_time.sum:.3f}s '
          f'BatchTime: {batch_time.sum:.3f}s '
          f'eta_time: {(args.max_epoch-epoch)*(batch_time.sum+data_time.sum)/3600:.2f}h '
          f'memory {memory_used:.0f}MB')

def test(net, queryloader, galleryloader):
    q_pool, q_feat, q_pids, q_camids = extract_feature(net, queryloader)
    g_pool, g_feat, g_pids, g_camids = extract_feature(net, galleryloader)
    q = F.normalize(q_feat, p=2, dim=1)
    g = F.normalize(g_feat, p=2, dim=1)
    distmat = -torch.mm(q, g.t()).numpy()
    cmc1, mAP1, mINP1 = eval(distmat, q_pids, g_pids, q_camids, g_camids)
    return cmc1, mAP1, mINP1

print('==> Start Training...', suffix, 'gpu:', os.environ['CUDA_VISIBLE_DEVICES'])
start_epoch = 1
best_r1 = -1
best_epoch = -1
for epoch in range(start_epoch, args.max_epoch + 1):
    train(epoch)
    if epoch % args.test_epoch == 0 or epoch == args.max_epoch:
        print(f'==> Start Testing Epoch: {epoch}')
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
                cmc_t, mAP_t, mINP_t = test(net, queryloader, galleryloader)
                if trial == 0:
                    cmc, mAP, mINP = cmc_t/t, mAP_t/t, mINP_t/t
                else:
                    cmc, mAP, mINP = cmc + cmc_t/t, mAP + mAP_t/t, mINP + mINP_t/t
            print(f'Rank-1: {cmc[0]:.2%} | Rank-5: {cmc[4]:.2%} | Rank-10: {cmc[9]:.2%} | '
                f'Rank-20: {cmc[19]:.2%} | mAP: {mAP:.2%} | mINP: {mINP:.2%} '
                f'Evaluation Time: {time.time() - start:.3f}')
            tmp_best_r1 = cmc[0]

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
            tmp_best_r1 = (tmp_best_r1 + cmc[0])/2.0
        
        if tmp_best_r1 > best_r1:
            if args.test == False:
                best_r1 = tmp_best_r1
                best_epoch = epoch
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc, 'mAP': mAP,
                    'mINP': mINP, 'epoch': epoch,
                }
                torch.save(state, checkpoint_path + 'model_best.t')
                # torch.save(state, checkpoint_path + 'epoch_{}.t'.format(epoch))

print(suffix)
print(f'start:{start_time}\n end :{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()+time_offset))}')
if args.test == False:
    log = f'{args.version:-3d}  Rank-1: {cmc[0]:.2%}  Rank-5: {cmc[4]:.2%}  Rank-10: {cmc[9]:.2%}  ' \
          f'Rank-20: {cmc[19]:.2%}  mAP: {mAP:.2%}  mINP:  {mINP:.2%}\n'
    with open(os.path.join(checkpoint_path, 'complete.txt'), 'w') as f:
        f.write(log)

if __name__ == '__main__':
    pass
