import torch

EPS = 1e-8


def masked_cross_entropy(class_dist, target, trg_mask):
    """
    :param class_dist: [batch_size, trg_seq_len, num_classes]
    :param target: [batch_size, trg_seq_len]
    :param trg_mask: [batch_size, trg_seq_len]
    :return:
    """
    num_classes = class_dist.size(2)
    class_dist_flat = class_dist.reshape(-1, num_classes)  # [batch_size*trg_seq_len, num_classes]
    target_flat = target.reshape(-1, 1)  # [batch*trg_seq_len, 1]
    log_dist_flat = torch.log(class_dist_flat + EPS)
    losses_flat = -torch.gather(log_dist_flat, dim=1, index=target_flat)  # [batch * trg_seq_len, 1]
    losses = losses_flat.view(*target.size())  # [batch, trg_seq_len]
    losses = losses * trg_mask
    loss = losses.sum()
    return loss