import torch


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    length: (batchsize,)
    """
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(*lengths.size(), 1)
            .lt(lengths.unsqueeze(-1)))


def pad(tensor, length):
    batch, max_sum_len = tensor.size()
    batch_, max_len1 = length.size()
    assert batch==batch_

    max_len2 = length.max()
    new_tensor = tensor.new_zeros(batch, max_len1, max_len2)
    for i in range(batch):
        len_ = 0
        for j, l in enumerate(length[i]):
            new_tensor[i][j][:l] = tensor[i][len_:len_+l]
            len_ += l
    return new_tensor
