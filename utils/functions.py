import time
import random
import numpy as np
import logging
import torch
import pykp.utils.io as io


def common_process_opt(opt):
    if opt.seed > 0:
        set_seed(opt.seed)

    return opt


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_since(start_time):
    return time.time() - start_time


def read_tokenized_src_file(path):
    """
    read tokenized source text file and convert them to list of list of words
    used for testing
    :param path:
    :return: data, a 2d list, each item in the list is a list of words of a src text, len(data) = num_lines
    """
    tokenized_train_src = []
    for line_idx, src_line in enumerate(open(path, 'r')):
        # process source line
        title_and_context = src_line.strip().split(io.EOS_WORD)
        if len(title_and_context) == 1:  # it only has context without title
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
        elif len(title_and_context) == 2:
            [title, context] = title_and_context
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            src_word_list = title_word_list + [io.TITLE_ABS_SEP] + context_word_list
        else:
            raise ValueError("The source text contains more than one title")
        # Append the lines to the data
        tokenized_train_src.append(src_word_list)
    return tokenized_train_src


def read_src_and_trg_files(src_file, trg_file, is_train):
    """
    used for training and evaluating
    """
    tokenized_train_src = []
    tokenized_train_trg = []
    filtered_cnt = 0
    for line_idx, (src_line, trg_line) in enumerate(zip(open(src_file, 'r'), open(trg_file, 'r'))):
        # process source line
        if (len(src_line.strip()) == 0) and is_train:
            continue
        title_and_context = src_line.strip().split(io.EOS_WORD)
        if len(title_and_context) == 1:  # it only has context without title
            [context] = title_and_context
            src_word_list = context.strip().split(' ')
        elif len(title_and_context) == 2:
            [title, context] = title_and_context
            title_word_list = title.strip().split(' ')
            context_word_list = context.strip().split(' ')
            src_word_list = title_word_list + [io.TITLE_ABS_SEP] + context_word_list
        else:
            raise ValueError("The source text contains more than one title")

        # process target line
        trg_list = trg_line.strip().split(';')  # a list of target sequences
        trg_word_list = [trg.split(' ') for trg in trg_list]
        # If it is training data, ignore the line with source length > 400 or target length > 60
        if is_train:
            if len(src_word_list) > 400 or len(trg_word_list) > 14:
                filtered_cnt += 1
                continue
        # Append the lines to the data
        tokenized_train_src.append(src_word_list)
        tokenized_train_trg.append(trg_word_list)

    assert len(tokenized_train_src) == len(
        tokenized_train_trg), 'the number of records in source and target are not the same'

    logging.info("%d rows filtered" % filtered_cnt)

    tokenized_train_pairs = list(zip(tokenized_train_src, tokenized_train_trg))
    return tokenized_train_pairs
