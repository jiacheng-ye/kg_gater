# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadGATLayer(nn.Module):
    def __init__(self, src_unit, tgt_unit, in_dim, out_dim, n_head, edge_embed_size, feat_drop, attn_drop,
                 ffn_hidden_size, ffn_drop):
        super(MultiHeadGATLayer, self).__init__()
        self.src_unit = src_unit
        self.tgt_unit = tgt_unit

        self.n_head = n_head
        self.out_dim = out_dim
        self.fc = nn.Linear(in_dim, n_head * out_dim, bias=False)
        self.feat_fc = nn.Linear(edge_embed_size, n_head * out_dim, bias=False)
        self.attn_fc = nn.Parameter(torch.FloatTensor(1, n_head, out_dim * 3))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.ffn = PositionwiseFeedForward(n_head * out_dim, ffn_hidden_size, ffn_drop)

        self.init_weight()

    def init_weight(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.feat_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc, gain=gain)

    def edge_attention(self, edges):
        dfeat = self.feat_fc(edges.data["embed"]).reshape(-1, self.n_head,
                                                               self.out_dim)  # [edge_num, n_head, out_dim]
        z2 = torch.cat([edges.src['z'], edges.dst['z'], dfeat], dim=-1)  # [edge_num, n_head, 3 * out_dim]
        z = F.leaky_relu((self.attn_fc * z2).sum(dim=-1, keepdim=True))  # [edge_num, n_head, 1]
        return {'e': z}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = self.attn_drop(F.softmax(nodes.mailbox['e'], dim=1))
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)  # (node_num, n_head, out_dim)
        return {'h': h}

    def forward(self, g, src_h, tgt_h):
        snode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == self.src_unit)
        tnode_id = g.filter_nodes(lambda nodes: nodes.data["unit"] == self.tgt_unit)
        stedge_id = g.filter_edges(lambda edges: (edges.src["unit"] == self.src_unit) & (edges.dst["unit"] == self.tgt_unit))

        src_h = self.feat_drop(src_h)
        z = self.fc(src_h).reshape(-1, self.n_head, self.out_dim)  # (node_num, n_head, out_dim)
        g.nodes[snode_id].data['z'] = z
        g.apply_edges(self.edge_attention, edges=stedge_id)
        g.pull(tnode_id, self.message_func, self.reduce_func)
        g.ndata.pop('z')
        h = g.ndata.pop('h')[tnode_id].reshape(-1, self.n_head * self.out_dim)

        h = F.elu(h) + tgt_h
        h = self.ffn(h.unsqueeze(0)).squeeze(0)
        return h


class GAT(nn.Module):
    def __init__(self, embeddings, edge_embed_size, input_size, n_head, ffn_hidden_size, feat_drop, attn_drop, ffn_drop,
                 n_iter):
        super(GAT, self).__init__()
        self.word_embed = embeddings
        self.w2d_edge_embed = nn.Embedding(10, edge_embed_size)  # box=10
        self.d2d_edge_embed = nn.Embedding(10, edge_embed_size)  # box=10
        self.n_iter = n_iter
        initrange = 0.1
        self.w2d_edge_embed.weight.data.uniform_(-initrange, initrange)
        self.d2d_edge_embed.weight.data.uniform_(-initrange, initrange)

        embed_size = self.word_embed.embedding_dim
        self.word2doc = MultiHeadGATLayer(src_unit=0,
                                            tgt_unit=1,
                                            in_dim=embed_size,
                                            out_dim=input_size//n_head,
                                            n_head=n_head,
                                            edge_embed_size=edge_embed_size,
                                            feat_drop=feat_drop,
                                            attn_drop=attn_drop,
                                            ffn_hidden_size=ffn_hidden_size,
                                            ffn_drop=ffn_drop
                                            )

        self.doc2word = MultiHeadGATLayer(src_unit=1,
                                            tgt_unit=0,
                                            in_dim=input_size,
                                            out_dim=embed_size//n_head,
                                            n_head=n_head,
                                            edge_embed_size=edge_embed_size,
                                            feat_drop=feat_drop,
                                            attn_drop=attn_drop,
                                            ffn_hidden_size=ffn_hidden_size,
                                            ffn_drop=ffn_drop
                                            )
        self.doc2doc = MultiHeadGATLayer(src_unit=1,
                                            tgt_unit=1,
                                            in_dim=input_size,
                                            out_dim=input_size // n_head,
                                            n_head=n_head,
                                            edge_embed_size=edge_embed_size,
                                            feat_drop=feat_drop,
                                            attn_drop=attn_drop,
                                            ffn_hidden_size=ffn_hidden_size,
                                            ffn_drop=ffn_drop
                                            )
    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(embeddings=embeddings,
                   edge_embed_size=opt.gat_edge_embed_size,
                   input_size=opt.d_model,
                   n_head=opt.gat_n_head,
                   ffn_hidden_size=opt.gat_ffn_hidden_size,
                   feat_drop=opt.gat_feat_drop,
                   attn_drop=opt.gat_atten_drop,
                   ffn_drop=opt.gat_ffn_drop,
                   n_iter=opt.gat_n_iter
                   )

    def init_graph(self, g, doc_rep):
        g = g.to(doc_rep.device)

        # w node
        word_nid = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 0)
        word_id = g.nodes[word_nid].data["id"]
        word_feat = self.word_embed(word_id.unsqueeze(1)).squeeze(1)
        g.nodes[word_nid].data["embed"] = word_feat

        # w2d edge
        edge_nid = g.filter_edges(lambda edges: edges.data["dtype"] == 0)
        edge_id = g.edges[edge_nid].data["score"]
        edge_feat = self.w2d_edge_embed(edge_id)
        g.edges[edge_nid].data["embed"] = edge_feat

        # d2d edge
        edge_nid = g.filter_edges(lambda edges: edges.data["dtype"] == 1)
        edge_id = g.edges[edge_nid].data["score"]
        edge_feat = self.d2d_edge_embed(edge_id)
        g.edges[edge_nid].data["embed"] = edge_feat

        # d node
        doc_nid = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        doc_idx = g.nodes[doc_nid].data["id"]
        doc_feat = doc_rep[doc_idx]
        g.nodes[doc_nid].data["init_feature"] = doc_feat

        return g

    def forward(self, graph, batch_docs, doc_rep):
        '''

        :param graph: [batch_size] * DGLGraph
        :param sent_rep: (batch_size, max(n_documents), max(sentences_per_document), input_size)
        :param doc_rep: (batch_size, max(n_documents), input_size)
        :param batch_docs: (batch_size, )
        :return:
            doc_rep: (batch_size, max(n_documents), input_size)
        '''
        graph = dgl.batch([self.init_graph(g, d) for g, d in zip(graph, doc_rep)])

        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        dnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 1)

        word_state = graph.nodes[wnode_id].data["embed"]
        doc_state = graph.nodes[dnode_id].data["init_feature"]
        for i in range(self.n_iter):
            word_state = self.doc2word(graph, doc_state, word_state)
            doc_state = self.word2doc(graph, word_state, doc_state)
            doc_state = self.doc2doc(graph, doc_state, doc_state)

        graph.nodes[dnode_id].data["hidden_state"] = doc_state

        doc_nid = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 2)
        doc_rep = graph.nodes[doc_nid].data["hidden_state"]
        doc_rep = stack_pad(doc_rep, batch_docs)

        return doc_rep


def stack_pad(tensor, length, pad_value=0):
    assert tensor.size(0) == sum(length)
    max_length = max(length)
    num = len(length)
    new_tensor = tensor.new_full((num, max_length, *tensor.size()[1:]), pad_value)
    index = 0
    for i, l in enumerate(length):
        new_tensor[i][:l] = tensor[index:index + l]
        index += l
    return new_tensor
