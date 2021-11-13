r"""
每个Decoder都有对应的State用来记录encoder的输出以及Decode的历史记录

"""

__all__ = [
    'State',
    "GRUState"
]

import torch


class State:
    def __init__(self, encoder_output=None, encoder_mask=None, **kwargs):
        """
        每个Decoder都有对应的State对象用来承载encoder的输出以及当前时刻之前的decode状态。

        :param Union[torch.Tensor, list, tuple] encoder_output: 如果不为None，内部元素需要为torch.Tensor, 默认其中第一维是batch
            维度
        :param Union[torch.Tensor, list, tuple] encoder_mask: 如果部位None，内部元素需要torch.Tensor, 默认其中第一维是batch
            维度
        :param kwargs:
        """
        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0

    @property
    def decode_length(self):
        """
        当前Decode到哪个token了，decoder只会从decode_length之后的token开始decode, 为0说明还没开始decode。

        :return:
        """
        return self._decode_length

    @decode_length.setter
    def decode_length(self, value):
        self._decode_length = value


class GRUState(State):
    def __init__(self, encoder_output, encoder_mask, hidden):
        """
        LSTMDecoder对应的State，保存encoder的输出以及LSTM解码过程中的一些中间状态

        :param torch.FloatTensor encoder_output: bsz x src_seq_len x encode_output_size，encoder的输出
        :param torch.BoolTensor encoder_mask: bsz x src_seq_len, 为0的地方是padding
        :param torch.FloatTensor hidden: num_layers x bsz x hidden_size, 上个时刻的hidden状态
        """
        super().__init__(encoder_output, encoder_mask)
        self.hidden = hidden
        self._input_feed = hidden[0]  # 默认是上一个时刻的输出

    @property
    def input_feed(self):
        """
        LSTMDecoder中每个时刻的输入会把上个token的embedding和input_feed拼接起来输入到下个时刻，在LSTMDecoder不使用attention时，
            input_feed即上个时刻的hidden state, 否则是attention layer的输出。
        :return: torch.FloatTensor, bsz x hidden_size
        """
        return self._input_feed

    @input_feed.setter
    def input_feed(self, value):
        self._input_feed = value

