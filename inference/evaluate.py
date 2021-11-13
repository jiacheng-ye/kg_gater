import logging
import os
import time
import torch
import pykp.utils.io as io
from pykp.utils.masked_loss import masked_cross_entropy
from utils.statistics import LossStatistics
from utils.string_helper import *
from utils.functions import time_since


def evaluate_loss(data_loader, model, opt):
    model.eval()
    evaluation_loss_sum = 0.0
    total_trg_tokens = 0
    n_batch = 0
    loss_compute_time_total = 0.0
    forward_time_total = 0.0

    with torch.no_grad():
        for batch_i, batch in enumerate(data_loader):
            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
            trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, _, \
            ref_docs, ref_lens, ref_doc_lens, ref_oovs, graph = batch

            max_num_oov = max([len(oov) for oov in oov_lists])  # max number of oov for each batch
            batch_size = src.size(0)
            n_batch += batch_size
            word2idx = opt.vocab['word2idx']
            target = trg_oov if opt.copy_attention else trg

            start_time = time.time()

            y_t_init = trg.new_ones(batch_size, 1) * word2idx[io.BOS_WORD]  # [batch_size, 1]
            input_tgt = torch.cat([y_t_init, trg[:, :-1]], dim=-1)
            memory_and_mask = model.encoder(src, src_lens, src_mask,
                                            ref_docs=ref_docs, ref_lens=ref_lens, ref_doc_lens=ref_doc_lens, graph=graph)
            state = model.decoder.init_state(*memory_and_mask)
            decoder_dist, attention_dist = model.decoder(input_tgt, state, src_oov, max_num_oov, ref_oovs=ref_oovs)

            forward_time = time_since(start_time)
            forward_time_total += forward_time

            start_time = time.time()
            loss = masked_cross_entropy(decoder_dist, target, trg_mask)
            loss_compute_time = time_since(start_time)
            loss_compute_time_total += loss_compute_time

            evaluation_loss_sum += loss.item()
            total_trg_tokens += trg_mask.sum().item()

    eval_loss_stat = LossStatistics(evaluation_loss_sum, total_trg_tokens, n_batch, forward_time=forward_time_total,
                                    loss_compute_time=loss_compute_time_total)
    return eval_loss_stat


def evaluate_greedy_generator(data_loader, generator, opt):
    pred_output_file = open(os.path.join(opt.pred_path, "predictions.txt"), "w")
    interval = 1000
    with torch.no_grad():
        word2idx = opt.vocab['word2idx']
        idx2word = opt.vocab['idx2word']
        start_time = time.time()
        for batch_i, batch in enumerate(data_loader):
            if (batch_i + 1) % interval == 0:
                logging.info("Batch %d: Time for running beam search on %d batches : %.1f" % (
                    batch_i + 1, interval, time_since(start_time)))
                start_time = time.time()

            src, src_lens, src_mask, src_oov, oov_lists, src_str_list, \
            trg_str_2dlist, trg, trg_oov, trg_lens, trg_mask, original_idx_list, \
            ref_docs, ref_lens, ref_doc_lens, ref_oovs, graph = batch

            n_best_result = generator.beam_search(src, src_lens, src_oov, src_mask, oov_lists, word2idx,
                                                  ref_docs, ref_lens, ref_doc_lens, ref_oovs, graph)
            pred_list = preprocess_n_best_result(n_best_result, idx2word, opt.vocab_size, oov_lists,
                                                 word2idx[io.EOS_WORD],
                                                 word2idx[io.UNK_WORD],
                                                 opt.replace_unk, src_str_list)

            # Process every src in the batch
            for src_str, trg_str_list, pred, oov in zip(src_str_list, trg_str_2dlist, pred_list, oov_lists):
                all_keyphrase_list = []
                for word_list in pred:
                    all_keyphrase_list += split_word_list_by_delimiter(word_list, io.SEP_WORD)
                write_example_kp(pred_output_file, all_keyphrase_list)

    pred_output_file.close()


def write_example_kp(out_file, kp_list):
    pred_print_out = ''
    for word_list_i, word_list in enumerate(kp_list):
        if word_list_i < len(kp_list) - 1:
            pred_print_out += '%s;' % ' '.join(word_list)
        else:
            pred_print_out += '%s' % ' '.join(word_list)
    pred_print_out += '\n'
    out_file.write(pred_print_out)


def preprocess_n_best_result(n_best_result, idx2word, vocab_size, oov_lists, eos_idx, unk_idx, replace_unk,
                             src_str_list):
    predictions = n_best_result['predictions']
    attention = n_best_result['attention']
    pred_list = []  # a list of dict, with len = batch_size
    for pred_n_best, attn_n_best, oov, src_word_list in zip(predictions, attention, oov_lists, src_str_list):
        sentences_n_best = []
        for pred, attn in zip(pred_n_best, attn_n_best):
            sentence = prediction_to_sentence(pred, idx2word, vocab_size, oov, eos_idx, unk_idx, replace_unk,
                                              src_word_list, attn)
            sentences_n_best.append(sentence)
        # a list of list of word, with len [n_best, out_seq_len], does not include tbe final <EOS>
        pred_list.append(sentences_n_best)
    return pred_list
