import torch.nn as nn

import pykp.utils.io as io
from pykp.decoder import *
from pykp.encoder import *


class Seq2SeqModel(nn.Module):
    """Container module with an encoder, decoder, embeddings."""

    def __init__(self, opt):
        """Initialize model."""
        super(Seq2SeqModel, self).__init__()
        embed = nn.Embedding(opt.vocab_size, opt.word_vec_size, opt.vocab["word2idx"][io.PAD_WORD])
        self.init_emb(embed)
        if opt.use_multidoc_graph:
            self.encoder = GraphRNNSeq2SeqEncoder.from_opt(opt, embed)
            self.decoder = GraphRNNSeq2SeqDecoder.from_opt(opt, embed)
        else:
            self.encoder = RNNSeq2SeqEncoder.from_opt(opt, embed)
            self.decoder = RNNSeq2SeqDecoder.from_opt(opt, embed)

    def init_emb(self, embed):
        """Initialize weights."""
        initrange = 0.1
        embed.weight.data.uniform_(-initrange, initrange)


