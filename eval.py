import torch
import numpy as np

def extract_feature(net, dataloader):
    net.eval()
    pools = torch.FloatTensor()
    feats = torch.FloatTensor()
    pid_s, camid_s = [], []
    with torch.no_grad():
        for imgs, pids, camids in dataloader:
            pid_s.extend(pids)
            camid_s.extend(camids)
            imgs = imgs.cuda()
            pool, feat = net(imgs)
            pools = torch.cat((pools, pool.cpu()), 0)
            feats = torch.cat((feats, feat.cpu()), 0)
    return pools, feats, np.asarray(pid_s), np.asarray(camid_s)

def extract_feature_infer(net, dataloader):
    net.eval()
    pools = torch.FloatTensor()
    feats = torch.FloatTensor()
    pid_s, camid_s, path_s = [], [], []
    with torch.no_grad():
        for imgs, pids, camids, paths in dataloader:
            pid_s.extend(pids)
            camid_s.extend(camids)
            path_s.extend(paths)
            imgs = imgs.cuda()
            pool, feat = net(imgs)
            pools = torch.cat((pools, pool.cpu()), 0)
            feats = torch.cat((feats, feat.cpu()), 0)
    return pools, feats, np.asarray(pid_s), np.asarray(camid_s), np.asarray(path_s)

def eval(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)#跨模态无同相机，sysu去除同场景，regdb不受此行代码影响
        keep = np.invert(remove)
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        cmc = orig_cmc.cumsum()
        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.
        # compute average precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP

if __name__ == '__main__':
    pass

