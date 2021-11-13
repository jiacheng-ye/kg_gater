# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.utils.data
from pykp.utils.functions import pad
from multiprocessing import Pool
from functools import partial

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
SEP_WORD = '<sep>'
DIGIT = '<digit>'
TITLE_ABS_SEP = '[SEP]'
PEOS_WORD = '<peos>'


class KeyphraseDataset(torch.utils.data.Dataset):
    def __init__(self, examples, word2idx, idx2word, device, load_train=True,
                 use_multidoc_graph=False, use_multidoc_copy=False):
        keys = ['src', 'src_oov', 'oov_dict', 'oov_list', 'src_str', 'trg_str', 'trg', 'trg_copy']
        if use_multidoc_graph:
            keys += ['ref_docs']
            keys += ['graph']
            if use_multidoc_copy:
                keys += ['ref_oovs']

        filtered_examples = []

        for e in examples:
            filtered_example = {}
            for k in keys:
                filtered_example[k] = e[k]
            if 'oov_list' in filtered_example:
                filtered_example['oov_number'] = len(filtered_example['oov_list'])
            filtered_examples.append(filtered_example)

        self.examples = filtered_examples
        self.word2idx = word2idx
        self.id2xword = idx2word
        self.load_train = load_train
        self.device = device
        self.use_multidoc_graph = use_multidoc_graph
        self.use_multidoc_copy = use_multidoc_copy

    @classmethod
    def build(cls, examples, opt, load_train):
        return cls(examples,
                   device=opt.device,
                   word2idx=opt.vocab['word2idx'],
                   idx2word=opt.vocab['idx2word'],
                   load_train=load_train,
                   use_multidoc_graph=opt.use_multidoc_graph,
                   use_multidoc_copy=opt.use_multidoc_copy)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _pad(self, input_list):
        input_list_lens = [len(l) for l in input_list]
        max_seq_len = max(input_list_lens)
        padded_batch = self.word2idx[PAD_WORD] * np.ones((len(input_list), max_seq_len))

        for j in range(len(input_list)):
            current_len = input_list_lens[j]
            padded_batch[j][:current_len] = input_list[j]

        padded_batch = torch.LongTensor(padded_batch)

        input_mask = torch.ne(padded_batch, self.word2idx[PAD_WORD]).type(torch.FloatTensor)

        return padded_batch, input_list_lens, input_mask

    def _pad2d(self, input_list):
        ''' list of list of 1d int'''
        batch_size = len(input_list)
        input_list_lens1 = [len(l) for l in input_list]

        max_seq_len1 = max(input_list_lens1)
        input_list_lens2 = self.word2idx[PAD_WORD] * np.ones((batch_size, max_seq_len1))

        max_seq_len = max([max([len(i) for i in l]) for l in input_list])
        padded_batch = self.word2idx[PAD_WORD] * np.ones((batch_size, max_seq_len1, max_seq_len))

        for i in range(batch_size):
            for j in range(len(input_list[i])):
                current_len = len(input_list[i][j])
                input_list_lens2[i][j] = current_len
                padded_batch[i][j][:current_len] = input_list[i][j]

        padded_batch = torch.LongTensor(padded_batch)
        input_list_lens1 = torch.LongTensor(input_list_lens1)
        input_list_lens2 = torch.LongTensor(input_list_lens2)
        return padded_batch, input_list_lens1, input_list_lens2

    def collate_fn_common(self, batches, trg=None, trg_oov=None):
        # source with oov words replaced by <unk>
        src = [b['src'] + [self.word2idx[EOS_WORD]] for b in batches]
        # extended src (oov words are replaced with temporary idx, e.g. 50000, 50001 etc.)
        src_oov = [b['src_oov'] + [self.word2idx[EOS_WORD]] for b in batches]

        oov_lists = [b['oov_list'] for b in batches]

        # b['src_str'] is a word_list for source text
        # b['trg_str'] is a list of word list
        src_str = [b['src_str'] for b in batches]
        trg_str = [b['trg_str'] for b in batches]

        batch_size = len(src)
        original_indices = list(range(batch_size))

        # pad the src and target sequences with <pad> token and convert to LongTensor
        src, src_lens, src_mask = self._pad(src)
        src_oov, _, _ = self._pad(src_oov)

        src = src.to(self.device)
        src_mask = src_mask.to(self.device)
        src_oov = src_oov.to(self.device)

        if self.load_train:
            trg, trg_lens, trg_mask = self._pad(trg)
            trg_oov, _, _ = self._pad(trg_oov)

            trg = trg.to(self.device)
            trg_mask = trg_mask.to(self.device)
            trg_oov = trg_oov.to(self.device)
        else:
            trg_lens, trg_mask = None, None

        if self.use_multidoc_graph:
            ref_docs = [b['ref_docs'] for b in batches]
            if self.use_multidoc_copy:
                assert batches[0]['ref_oovs'] is not None, "set use_multidoc_copy in preprocess!"
                ref_oovs = [b['ref_oovs'] for b in batches]
            else:
                ref_oovs = None
            from retrievers.utils import build_graph
            graph = [build_graph(**b['graph']) for b in batches]

            ref_docs, ref_lens, ref_doc_lens = self._pad2d(ref_docs)
            ref_docs = ref_docs.to(self.device)
            ref_lens = ref_lens.to(self.device)
            ref_doc_lens = ref_doc_lens.to(self.device)

            if self.use_multidoc_copy:
                ref_oovs, _, _ = self._pad(ref_oovs)
                ref_oovs = ref_oovs.to(self.device)
                ref_oovs = pad(ref_oovs, ref_doc_lens)
        else:
            ref_docs, ref_oovs, graph, ref_lens, ref_doc_lens = None, None, None, None, None

        return src, src_lens, src_mask, src_oov, oov_lists, src_str, \
               trg_str, trg, trg_oov, trg_lens, trg_mask, original_indices, \
               ref_docs, ref_lens, ref_doc_lens, ref_oovs, graph

    def collate_fn_one2one(self, batches):
        if self.load_train:
            trg = [b['trg'] + [self.word2idx[EOS_WORD]] for b in batches]
            trg_oov = [b['trg_copy'] + [self.word2idx[EOS_WORD]] for b in batches]
            return self.collate_fn_common(batches, trg, trg_oov)
        else:
            return self.collate_fn_common(batches)

    def collate_fn_one2seq(self, batches):
        if self.load_train:
            trg = []
            trg_oov = []
            for b in batches:
                trg_concat = []
                trg_oov_concat = []
                trg_size = len(b['trg'])
                assert len(b['trg']) == len(b['trg_copy'])
                for trg_idx, (trg_phase, trg_phase_oov) in enumerate(zip(b['trg'], b['trg_copy'])):
                    # ignore the <peos> word if it exists
                    if self.word2idx[PEOS_WORD] in trg_phase:
                        continue
                    # if this is the last keyphrase, end with <eos>
                    if trg_idx == trg_size - 1:
                        trg_concat += trg_phase + [self.word2idx[EOS_WORD]]
                        trg_oov_concat += trg_phase_oov + [self.word2idx[EOS_WORD]]
                    else:
                        # trg_concat = [target_1] + [sep] + [target_2] + [sep] + ...
                        trg_concat += trg_phase + [self.word2idx[SEP_WORD]]
                        trg_oov_concat += trg_phase_oov + [self.word2idx[SEP_WORD]]
                trg.append(trg_concat)
                trg_oov.append(trg_oov_concat)

            return self.collate_fn_common(batches, trg, trg_oov)
        else:
            return self.collate_fn_common(batches)


def build_interactive_predict_dataset(tokenized_src, opt, mode="one2many", include_original=True):
    # build a dummy trg list, and then combine it with src, and pass it to the build_dataset method
    num_lines = len(tokenized_src)
    tokenized_trg = [['.']] * num_lines  # create a dummy tokenized_trg
    tokenized_src_trg_pairs = list(zip(tokenized_src, tokenized_trg))
    return build_dataset(tokenized_src_trg_pairs, opt, mode=mode, include_original=include_original, is_train=False)


def build_one_example(src_tgt_pair, ref_docs_tokenized=None, graph_utils=None, opt=None, mode='one2one', include_original=False, is_train=True):

    word2idx = opt.vocab['word2idx']
    # if w's id is larger than opt.vocab_size, replace with <unk>
    source, targets = src_tgt_pair
    src = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in source]

    src_oov, oov_dict, oov_list = extend_vocab_OOV(source, word2idx, opt.vocab_size, opt.max_unk_words)

    if opt.retriever is not None:
        if ref_docs_tokenized is None:
            source_str = ' '.join(source)
            ref_docs_tokenized, graph_utils = opt.retriever.maybe_retrieving_building_graph(
                source_str, word2idx, vocab_size=opt.vocab_size, is_train=is_train)
        ref_docs = [[word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in doc]
                    for doc in ref_docs_tokenized]
        if opt.use_multidoc_copy:
            flattened_ref = []
            for ref in ref_docs_tokenized:
                flattened_ref += ref
            ref_oovs, oov_dict, oov_list = extend_vocab_OOV(flattened_ref, word2idx, opt.vocab_size,
                                                               opt.max_unk_words, oov_dict)
        else:
            ref_oovs = None
    else:
        ref_docs, ref_oovs, graph_utils = None, None, None

    examples = []  # for one-to-many
    for target in targets:
        example = {}
        if opt.retriever is not None:
            example['ref_docs'] = ref_docs
            if graph_utils:
                example['graph'] = graph_utils
            if opt.use_multidoc_copy:
                example['ref_oovs'] = ref_oovs

        if include_original:
            example['src_str'] = source
            example['trg_str'] = target

        example['src'] = src
        trg = [word2idx[w] if w in word2idx and word2idx[w] < opt.vocab_size else word2idx[UNK_WORD] for w in target]
        example['trg'] = trg
        example['src_oov'] = src_oov
        example['oov_dict'] = oov_dict
        example['oov_list'] = oov_list

        # oov words are replaced with new index
        trg_copy = []
        for w in target:
            if w in word2idx and word2idx[w] < opt.vocab_size:
                trg_copy.append(word2idx[w])
            elif w in oov_dict:
                trg_copy.append(oov_dict[w])
            else:
                trg_copy.append(word2idx[UNK_WORD])
        example['trg_copy'] = trg_copy
        examples.append(example)

    if mode == 'one2many' and len(examples) > 0:
        o2m_example = {}
        keys = examples[0].keys()
        for key in keys:
            if key.startswith('src') or key.startswith('oov') \
                    or key.startswith('ref') or key.startswith('graph'):
                o2m_example[key] = examples[0][key]
            else:
                o2m_example[key] = [e[key] for e in examples]

        if include_original:
            assert len(o2m_example['src']) == len(o2m_example['src_oov']) == len(o2m_example['src_str'])
            assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
            assert len(o2m_example['trg']) == len(o2m_example['trg_copy']) == len(o2m_example['trg_str'])
        else:
            assert len(o2m_example['src']) == len(o2m_example['src_oov'])
            assert len(o2m_example['oov_dict']) == len(o2m_example['oov_list'])
            assert len(o2m_example['trg']) == len(o2m_example['trg_copy'])
        return o2m_example
    else:
        return examples


def build_dataset(src_trgs_pairs, opt, mode='one2one', include_original=True, is_train=True):
    '''
    Standard process for copy model
    :param mode: one2one or one2many
    :param include_original: keep the original texts of source and target
    :return:
    '''
    _build_one_example = partial(build_one_example, opt=opt, mode=mode, include_original=include_original,
                                 is_train=is_train)
    if not opt.retriever or not opt.dense_retrieve:
        with Pool(opt.num_workers) as processes:
            examples = processes.starmap(_build_one_example, zip(src_trgs_pairs))
    else:
        ref_docs_tokenized, graph_utils = opt.retriever.batch_maybe_retrieving_building_graph(
            [' '.join(pair[0]) for pair in src_trgs_pairs], opt.vocab['word2idx'],
            vocab_size=opt.vocab_size, is_train=is_train)
        if graph_utils is None:
            graph_utils = [None] * len(src_trgs_pairs)
        examples = [_build_one_example(i, j, k) for i, j, k in zip(src_trgs_pairs, ref_docs_tokenized, graph_utils)]

    if mode == 'one2one':
        return_examples = []
        for exps in examples:
            for ex in exps:
                return_examples.append(ex)
    else:
        return_examples = examples

    return return_examples


def extend_vocab_OOV(source_words, word2idx, vocab_size, max_unk_words, pre_oov_dict=None):
    """
    Map source words to their ids, including OOV words. Also return a list of OOVs in the article.
    if the number of oovs in the source text is more than max_unk_words, ignore and replace them as <unk>
    """
    src_oov = []
    if pre_oov_dict is None:
        oov_dict = {}
    else:
        oov_dict = pre_oov_dict
    for w in source_words:
        if w in word2idx and word2idx[w] < vocab_size:  # a OOV can be either outside the vocab or id>=vocab_size
            src_oov.append(word2idx[w])
        elif w in oov_dict:
            src_oov.append(oov_dict[w])
        else:
            if len(oov_dict) < max_unk_words:
                # e.g. 50000 for the first article OOV, 50001 for the second...
                word_id = len(oov_dict) + vocab_size
                oov_dict[w] = word_id
                src_oov.append(word_id)
            else:
                # exceeds the maximum number of acceptable oov words, replace it with <unk>
                word_id = word2idx[UNK_WORD]
                src_oov.append(word_id)

    oov_list = [w for w, w_id in sorted(oov_dict.items(), key=lambda x: x[1])]
    return src_oov, oov_dict, oov_list
