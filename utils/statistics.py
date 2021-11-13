import math


class LossStatistics:
    """
    Accumulator for loss staistics. Modified from OpenNMT
    """

    def __init__(self, loss=0.0, n_tokens=0, n_batch=0, forward_time=0.0, loss_compute_time=0.0, backward_time=0.0):
        self.loss = loss
        if math.isnan(loss):
            raise ValueError("Loss is NaN")
        self.n_tokens = n_tokens
        self.n_batch = n_batch
        self.forward_time = forward_time
        self.loss_compute_time = loss_compute_time
        self.backward_time = backward_time

    def update(self, stat):
        """
        Update statistics by suming values with another `LossStatistics` object

        Args:
            stat: another statistic object
        """
        self.loss += stat.loss
        if math.isnan(stat.loss):
            raise ValueError("Loss is NaN")
        self.n_tokens += stat.n_tokens
        self.n_batch += stat.n_batch
        self.forward_time += stat.forward_time
        self.loss_compute_time += stat.loss_compute_time
        self.backward_time += stat.backward_time

    def xent(self):
        """ compute normalized cross entropy """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return self.loss / self.n_tokens

    def ppl(self):
        """ compute normalized perplexity """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return math.exp(min(self.loss / self.n_tokens, 100))

    def total_time(self):
        return self.forward_time, self.loss_compute_time, self.backward_time

    def clear(self):
        self.loss = 0.0
        self.n_tokens = 0
        self.n_batch = 0
        self.forward_time = 0.0
        self.loss_compute_time = 0.0
        self.backward_time = 0.0
