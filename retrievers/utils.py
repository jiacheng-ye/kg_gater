import dgl
import torch
import pykp.utils.io as io


def read_tokenized_src_file(src_file):
    references = []
    filtered_cnt = 0
    for src_line in open(src_file, 'r'):
        if (len(src_line.strip()) == 0):
            continue
        title_and_context = src_line.strip().split(io.EOS_WORD)
        [title, context] = title_and_context

        title_word_list = title.strip().split(' ')
        context_word_list = context.strip().split(' ')
        src_word_list = title_word_list + [io.TITLE_ABS_SEP] + context_word_list
        if len(src_word_list) > 400:
            filtered_cnt += 1
            continue
        references.append(' '.join(src_word_list))
    print("%d/%d rows filtered" % (filtered_cnt, len(references)))
    return references


def read_src_and_trg_files(src_file, trg_file, use_doc=True, use_kp=True):
    references_with_trg = []
    filtered_cnt = 0
    for line_idx, (src_line, trg_line) in enumerate(zip(open(src_file, 'r'), open(trg_file, 'r'))):
        # process source line
        if (len(src_line.strip()) == 0):
            continue
        title_and_context = src_line.strip().split(io.EOS_WORD)
        [title, context] = title_and_context

        title_word_list = title.strip().split(' ')
        context_word_list = context.strip().split(' ')
        src_word_list = title_word_list + [io.TITLE_ABS_SEP] + context_word_list

        if len(src_word_list) > 400:
            filtered_cnt += 1
            continue

        # process target line
        trg_list = trg_line.strip().split(';')  # a list of target sequences

        # Append the lines to the data
        if use_kp:
            kp_str = ' {0} '.format(io.SEP_WORD).join(trg_list)

            if use_doc:
                line = ' '.join(src_word_list) + ' ' + kp_str
            else:
                line = kp_str
        elif use_doc:
            line = ' '.join(src_word_list)
        else:
            raise Exception('One of use_doc or use_kp must be set!')

        references_with_trg.append(line)

    print("%d/%d rows filtered" % (filtered_cnt, len(references_with_trg)))

    return references_with_trg


def get_node_edge(docs, w2d_score, d2d_score):
    """

    :param docs:
    :param w2d_score:
    :param d2d_score:
    :return:
    """
    wid2wnid = {}
    w_id, d_idx = [], []
    w_d_wnid, w_d_dnid = [], []
    w_d_feat = {"score": [], 'dtype': []}
    d_d_dnid1, d_d_dnid2 = [], []
    d_d_feat = {"score": [], 'dtype': []}
    wnid, dnid = 0, 0
    for di, (_, wd_scores) in enumerate(zip(docs, w2d_score)):
        for wid, wd_score in wd_scores.items():
            if wid not in wid2wnid:
                wid2wnid[wid] = wnid
                w_id.append(wid)
                wnid += 1
            w_d_wnid.append(wid2wnid[wid])
            w_d_dnid.append(dnid)
            w_d_feat["score"].append(wd_score)
            w_d_feat["dtype"].append(0)
        d_idx.append(di)
        dnid += 1

        if di > 0:
            d_d_dnid1.append(0)
            d_d_dnid2.append(di)
            # d_d_feat["score"].append(max(0, min(round(math.log(d2d_score[di-1])), 9)))
            d_d_feat["score"].append(d2d_score[di - 1])
            d_d_feat["dtype"].append(1)


    utils = {"w_id": w_id,
             "d_idx": d_idx,
             "w_d_wnid": w_d_wnid,
             "w_d_dnid": w_d_dnid,
             "w_d_feat": w_d_feat,
             "d_d_dnid1": d_d_dnid1,
             "d_d_dnid2": d_d_dnid2,
             "d_d_feat": d_d_feat
             }

    return utils


def build_graph(w_id, d_idx, w_d_wnid, w_d_dnid, w_d_feat, d_d_dnid1, d_d_dnid2, d_d_feat):
    G = dgl.DGLGraph()
    w_num = len(w_id)
    d_num = len(d_idx)

    # add word nodes
    G.add_nodes(w_num)
    G.ndata["unit"] = torch.zeros(w_num)
    G.ndata["dtype"] = torch.zeros(w_num)
    G.ndata["id"] = torch.LongTensor(w_id)

    # add doc nodes
    G.add_nodes(d_num)
    G.ndata["unit"][w_num:] = torch.ones(d_num)
    G.ndata["dtype"][w_num:] = torch.ones(d_num) * 2
    G.ndata["id"][w_num:] = torch.LongTensor(d_idx)

    # add w2d edges
    w_d_wnid = torch.LongTensor(w_d_wnid)
    w_d_dnid = torch.LongTensor(w_d_dnid) + w_num
    w_d_feat["score"] = torch.LongTensor(w_d_feat["score"])
    w_d_feat["dtype"] = torch.LongTensor(w_d_feat["dtype"])
    G.add_edges(w_d_wnid, w_d_dnid, w_d_feat)
    G.add_edges(w_d_dnid, w_d_wnid, w_d_feat)

    # add d2d edges
    d_d_dnid1 = torch.LongTensor(d_d_dnid1) + w_num
    d_d_dnid2 = torch.LongTensor(d_d_dnid2) + w_num
    d_d_feat["score"] = torch.LongTensor(d_d_feat["score"])
    d_d_feat["dtype"] = torch.LongTensor(d_d_feat["dtype"])
    G.add_edges(d_d_dnid1, d_d_dnid2, d_d_feat)
    G.add_edges(d_d_dnid2, d_d_dnid1, d_d_feat)

    return G
