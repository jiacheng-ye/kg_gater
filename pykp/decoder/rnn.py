import torch.nn as nn
import torch
from pykp.utils.seq2seq_state import GRUState
from pykp.modules.attention import Attention, HierAttention
import torch.nn.functional as F


class RNNSeq2SeqDecoder(nn.Module):
    def __init__(self, embed, num_layers=3, hidden_size=300, dropout=0.3, copy_attn=False):
        """
        LSTM的Decoder
        :param nn.Module,tuple embed: decoder输入的embedding.
        :param int num_layers: 多少层LSTM
        :param int hidden_size: 隐藏层大小, 该值也被认为是encoder的输出维度大小
        :param dropout: Dropout的大小
        """
        super(RNNSeq2SeqDecoder, self).__init__()
        self.embed = embed
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size=self.embed.embedding_dim, hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True, bidirectional=False, dropout=dropout if num_layers > 1 else 0)

        self.attention_layer = Attention(hidden_size, hidden_size)

        self.vocab_size = embed.num_embeddings
        self.output_fc = nn.Linear(hidden_size*2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.copy_attn = copy_attn
        if copy_attn:
            p_gen_input_size = self.embed.embedding_dim + hidden_size + hidden_size
            self.p_gen_linear = nn.Linear(p_gen_input_size, 2)

    @classmethod
    def from_opt(cls, opt, embed):
        return cls(embed,
                   num_layers=opt.dec_layers,
                   hidden_size=opt.d_model,
                   dropout=opt.dropout,
                   copy_attn=opt.copy_attention)

    def forward(self, tokens, state, src_oov, max_num_oov, **kwargs):
        """
        :param torch.LongTensor tokens: batch x max_len
        :param LSTMState state: 保存encoder输出和decode状态的State对象
        :param bool return_attention: 是否返回attention的的score
        :return: bsz x max_len x vocab_size; 如果return_attention=True, 还会返回bsz x max_len x encode_length
        """

        src_output = state.encoder_output[0]
        encoder_mask = state.encoder_mask

        assert state.decode_length < tokens.size(1), "The decoded tokens in State should be less than tokens."
        tokens = tokens[:, state.decode_length:]
        batch_size, trg_len = tokens.size()

        x = self.embed(tokens)

        attn_dist = []
        contexts = []
        hiddens = []
        cur_hidden = state.hidden
        for i in range(trg_len):
            _, cur_hidden = self.rnn(x[:, i:i + 1, :], hx=cur_hidden)
            context, attn_weight = self.attention_layer(cur_hidden[-1], src_output, encoder_mask)

            state.hidden = cur_hidden
            state.decode_length += 1
            contexts.append(context)
            hiddens.append(cur_hidden[-1])
            attn_dist.append(attn_weight)

        attn_dist = torch.stack(attn_dist, dim=1)  # batch, tgt_len, src_len
        contexts = torch.stack(contexts, dim=1)  # batch,seq_len,hidden
        hiddens = torch.stack(hiddens, dim=1)  # batch,seq_len,hidden
        vocab_dist = F.softmax(self.output_layer(self.dropout_layer(self.output_fc(torch.cat([contexts, hiddens], -1)
                                                                                   ))), -1)

        if self.copy_attn:
            p_gen_input = torch.cat([contexts, hiddens, x], dim=-1)
            p_gen_dist = self.p_gen_linear(p_gen_input).softmax(-1)
            vocab_dist_ = p_gen_dist[:, :, 0].unsqueeze(-1) * vocab_dist
            attn_dist_ = p_gen_dist[:, :, 1].unsqueeze(-1) * attn_dist

            if max_num_oov > 0:
                extra_zeros = vocab_dist_.new_zeros((batch_size, trg_len, max_num_oov))
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=-1)

            final_dist = vocab_dist_.scatter_add(2, src_oov[:, None, :].repeat(1, trg_len, 1), attn_dist_)
            assert final_dist.size() == torch.Size([batch_size, trg_len, self.vocab_size + max_num_oov])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, trg_len, self.vocab_size])

        return final_dist, attn_dist

    def init_state(self, encoder_output, encoder_mask):
        cur_doc_rep = encoder_output[1]

        assert cur_doc_rep.dim() == 2
        assert cur_doc_rep.size(-1) == self.hidden_size
        hidden = cur_doc_rep[None].repeat(self.num_layers, 1, 1)  # num_layers x bsz x hidden_size

        state = GRUState(encoder_output, encoder_mask, hidden)

        return state


class GraphRNNSeq2SeqDecoder(RNNSeq2SeqDecoder):
    def __init__(self, embed, num_layers=3, hidden_size=300, dropout=0.3, copy_attn=False,
                 use_multidoc_copy=False):
        """
        LSTM的Decoder
        :param nn.Module,tuple embed: decoder输入的embedding.
        :param int num_layers: 多少层LSTM
        :param int hidden_size: 隐藏层大小, 该值也被认为是encoder的输出维度大小
        :param dropout: Dropout的大小
        """
        super(GraphRNNSeq2SeqDecoder, self).__init__(embed, num_layers, hidden_size, dropout, copy_attn)
        self.ref_attention_layer = HierAttention(hidden_size, hidden_size)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )
        self.use_multidoc_copy = use_multidoc_copy
        if copy_attn:
            p_gen_input_size = self.embed.embedding_dim + hidden_size + hidden_size
            if self.use_multidoc_copy:
                self.p_gen_linear = nn.Linear(p_gen_input_size, 3)
            else:
                self.p_gen_linear = nn.Linear(p_gen_input_size, 2)

    @classmethod
    def from_opt(cls, opt, embed):
        return cls(embed,
                   num_layers=opt.dec_layers,
                   hidden_size=opt.d_model,
                   dropout=opt.dropout,
                   copy_attn=opt.copy_attention,
                   use_multidoc_copy=opt.use_multidoc_copy
                   )

    def forward(self, tokens, state, src_oov, max_num_oov, ref_oovs):
        """
        :param torch.LongTensor tokens: batch x max_len
        :param LSTMState state: 保存encoder输出和decode状态的State对象
        :param bool return_attention: 是否返回attention的的score
        :return: bsz x max_len x vocab_size; 如果return_attention=True, 还会返回bsz x max_len x encode_length
        """
        cur_word_rep, cur_doc_rep, ref_word_reps, ref_doc_reps = state.encoder_output
        src_mask, ref_word_mask, ref_doc_mask = state.encoder_mask

        assert state.decode_length < tokens.size(1), "The decoded tokens in State should be less than tokens."
        tokens = tokens[:, state.decode_length:]
        batch_size, trg_len = tokens.size()

        x = self.embed(tokens)

        attn_dists = [] if self.attention_layer is not None else None  # 保存attention weight, batch,tgt_seq,src_seq
        ref_attn_dists = []
        contexts = []
        hiddens = []
        cur_hidden = state.hidden
        for i in range(trg_len):
            _, cur_hidden = self.rnn(x[:, i:i + 1, :], hx=cur_hidden)
            cur_context, attn_dist = self.attention_layer(cur_hidden[-1], cur_word_rep, src_mask)
            ref_context, ref_attn_dist = self.ref_attention_layer(cur_hidden[-1], ref_doc_reps,
                                                                  ref_word_reps, ref_doc_mask,
                                                                  ref_word_mask)
            gate = self.fusion_gate(torch.cat([cur_context, ref_context], -1))
            context = gate * cur_context + (1 - gate) * ref_context

            state.hidden = cur_hidden
            state.decode_length += 1
            contexts.append(context)
            hiddens.append(cur_hidden[-1])
            attn_dists.append(attn_dist)
            ref_attn_dists.append(ref_attn_dist)

        if attn_dists is not None:
            attn_dists = torch.stack(attn_dists, dim=1)  # batch, tgt_len, src_len
        if ref_attn_dists is not None:
            ref_attn_dists = torch.stack(ref_attn_dists, dim=1)  # batch, tgt_len, doc, src_len

        contexts = torch.stack(contexts, dim=1)  # batch,seq_len,hidden
        hiddens = torch.stack(hiddens, dim=1)  # batch,seq_len,hidden
        vocab_dist = F.softmax(self.output_layer(self.dropout_layer(self.output_fc(torch.cat([contexts, hiddens], -1)
                                                                                   ))), -1)

        if self.copy_attn:
            p_gen_input = torch.cat([contexts, hiddens, x], dim=-1)
            p_gen_dist = self.p_gen_linear(p_gen_input).softmax(-1)
            vocab_dist_ = p_gen_dist[:, :, 0].unsqueeze(-1) * vocab_dist
            attn_dist_ = p_gen_dist[:, :, 1].unsqueeze(-1) * attn_dists
            if max_num_oov > 0:
                extra_zeros = vocab_dist_.new_zeros((batch_size, trg_len, max_num_oov))
                vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=-1)
            src_oov = src_oov[:, None, :].repeat(1, trg_len, 1)
            assert src_oov.size() == attn_dists.size()
            final_dist = vocab_dist_.scatter_add(2, src_oov, attn_dist_)

            if self.use_multidoc_copy:
                ref_oovs = ref_oovs[:, None, :, :].repeat(1, trg_len, 1, 1)
                assert ref_oovs.size() == ref_attn_dists.size()
                ref_attn_dist_ = p_gen_dist[:, :, 2].unsqueeze(-1).unsqueeze(-1) * ref_attn_dists
                final_dist = final_dist.scatter_add(2, ref_oovs.reshape(batch_size, trg_len, -1),
                                                    ref_attn_dist_.reshape(batch_size, trg_len, -1))

            assert final_dist.size() == torch.Size([batch_size, trg_len, self.vocab_size + max_num_oov])
        else:
            final_dist = vocab_dist
            assert final_dist.size() == torch.Size([batch_size, trg_len, self.vocab_size])

        return final_dist, attn_dists

