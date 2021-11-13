import torch.nn as nn
import torch
from pykp.encoder.gat import GAT
from pykp.utils.functions import sequence_mask


class RNNSeq2SeqEncoder(nn.Module):
    def __init__(self, embed, num_layers=3, hidden_size=400, dropout=0.3):
        """
        LSTM的Encoder
        :param embed: encoder的token embed
        :param int num_layers: 多少层
        :param int hidden_size: LSTM隐藏层、输出的大小
        :param float dropout: LSTM层之间的Dropout是多少
        """
        super(RNNSeq2SeqEncoder, self).__init__()
        self.embed = embed
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.cur_rnn = nn.GRU(input_size=embed.embedding_dim, hidden_size=hidden_size // 2, num_layers=num_layers,
                              bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)

    @classmethod
    def from_opt(cls, opt, embed):
        return cls(embed,
                   num_layers=opt.enc_layers,
                   hidden_size=opt.d_model,
                   dropout=opt.dropout)

    def _forward(self, rnn, src, src_lens):
        src_embed = self.embed(src)  # [batch, src_len, embed_size]
        packed_input_src = nn.utils.rnn.pack_padded_sequence(src_embed, src_lens, batch_first=True,
                                                             enforce_sorted=False)
        word_rep, doc_rep = rnn(packed_input_src)
        # word_rep [batch, seq_len, num_directions*hidden_size]
        # doc_rep [num_layer * num_directions, batch, hidden_size]
        word_rep, _ = nn.utils.rnn.pad_packed_sequence(word_rep, batch_first=True)  # unpack (back to padded)

        # only extract the final state in the last layer
        doc_rep = torch.cat((doc_rep[-1, :, :], doc_rep[-2, :, :]), 1)  # [batch, hidden_size*2]
        return word_rep, doc_rep

    def forward(self, src, src_lens, src_mask=None, **kwargs):
        """
        :param src: [batch, src_seq_len]
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch
        Other parameters will not be used in the RNNENcoderBasic class, they are here because we want to have a unify interface
        :return:
        """
        word_rep, doc_rep = self._forward(self.cur_rnn, src, src_lens)
        return (word_rep, doc_rep), src_mask


class GraphRNNSeq2SeqEncoder(RNNSeq2SeqEncoder):
    def __init__(self, embed, num_layers=3, hidden_size=400, dropout=0.3, gat=None):
        super(GraphRNNSeq2SeqEncoder, self).__init__(embed, num_layers, hidden_size, dropout)
        self.ref_rnn = nn.GRU(input_size=embed.embedding_dim, hidden_size=hidden_size // 2, num_layers=num_layers+1,
                              bidirectional=True, batch_first=True, dropout=dropout)
        self.gat = gat

    @classmethod
    def from_opt(cls, opt, embed):
        return cls(embed,
                   num_layers=opt.enc_layers,
                   hidden_size=opt.d_model,
                   dropout=opt.dropout,
                   gat=GAT.from_opt(opt, embed))

    def forward(self, src, src_lens, src_mask=None, ref_docs=None, ref_lens=None, ref_doc_lens=None, graph=None):
        cur_word_rep, cur_doc_rep = self._forward(self.cur_rnn, src, src_lens)
        packed_ref_docs_by_ref_lens = nn.utils.rnn.pack_padded_sequence(ref_docs, ref_lens.cpu(), batch_first=True,
                                                                        enforce_sorted=False)
        packed_doc_ref_lens_by_ref_lens = nn.utils.rnn.pack_padded_sequence(ref_doc_lens, ref_lens.cpu(),
                                                                            batch_first=True, enforce_sorted=False)

        packed_ref_word_reps, packed_ref_doc_reps = self._forward(self.ref_rnn, packed_ref_docs_by_ref_lens.data,
                                                                  packed_doc_ref_lens_by_ref_lens.data.cpu())

        ref_word_reps, _ = nn.utils.rnn.pad_packed_sequence(
            nn.utils.rnn.PackedSequence(data=packed_ref_word_reps,
                                        batch_sizes=packed_ref_docs_by_ref_lens.batch_sizes,
                                        sorted_indices=packed_ref_docs_by_ref_lens.sorted_indices,
                                        unsorted_indices=packed_ref_docs_by_ref_lens.unsorted_indices),
            batch_first=True
        )  # [batch, max_doc_num, max_doc_len, hidden_size]
        ref_doc_reps, _ = nn.utils.rnn.pad_packed_sequence(
            nn.utils.rnn.PackedSequence(data=packed_ref_doc_reps,
                                        batch_sizes=packed_ref_docs_by_ref_lens.batch_sizes,
                                        sorted_indices=packed_ref_docs_by_ref_lens.sorted_indices,
                                        unsorted_indices=packed_ref_docs_by_ref_lens.unsorted_indices),
            batch_first=True
        )  # [batch, max_doc_num, hidden_size]
        all_doc_rep = torch.cat([cur_doc_rep.unsqueeze(1), ref_doc_reps], 1)

        assert graph is not None
        all_doc_rep = self.gat(graph, ref_lens + 1, all_doc_rep)
        cur_doc_rep = all_doc_rep[:, 0].contiguous()
        ref_doc_reps = all_doc_rep[:, 1:].contiguous()

        ref_doc_mask = sequence_mask(ref_lens)
        ref_word_mask = sequence_mask(ref_doc_lens)
        return (cur_word_rep, cur_doc_rep, ref_word_reps, ref_doc_reps), (src_mask, ref_word_mask, ref_doc_mask)
