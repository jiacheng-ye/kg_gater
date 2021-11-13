#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool
from functools import partial
import math
import retrievers.ranker_utils as utils
import retrievers.tokenizer as tokenizers
import string

logger = logging.getLogger(__name__)
stoplist = list(string.punctuation)


class TfidfDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, tfidf_path=None, strict=False):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        logger.info('Loading %s' % tfidf_path)
        matrix, metadata = utils.load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]

        # normalize score to integer between [0, 9]
        doc_scores = [max(0, min(9, round(math.log(d2d_score)))) for d2d_score in doc_scores]

        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None, query_ids=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain processes here as scipy is outside of the GIL.
        """
        with Pool(num_workers) as processes:
            closest_docs = partial(self.closest_docs, k=k)
            if query_ids:
                results = processes.starmap(closest_docs, zip(queries, query_ids))
            else:
                results = processes.map(closest_docs, queries)
        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.

        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        data = np.multiply(tfs, idfs)

        # One row, sparse csr matrix
        indptr = np.array([0, len(wids_unique)])
        spvec = sp.csr_matrix(
            (data, wids_unique, indptr), shape=(1, self.hash_size)
        )

        return spvec

    def batch_words_tfidf(self, queries, k=20, num_workers=None):
        with Pool(num_workers) as processes:
            words_tfidf = partial(self.words_tfidf, k=k)
            results = processes.map(words_tfidf, queries)
        return results

    def words_tfidf(self, query, k=20, word2idx=None, vocab_size=50002):
        tokens = self.tokenizer.tokenize(utils.normalize(query))
        words = tokens.ngrams(n=1, uncased=True,filter_fn=utils.filter_ngram)
        wids2words = {utils.hash(w, self.hash_size): w for w in words}
        if len(wids2words) == 0:
            return {}

        # Count TF
        wids_unique, wids_counts = np.unique(list(wids2words.keys()), return_counts=True)
        tfs = np.log1p(wids_counts)

        # Count IDF
        Ns = self.doc_freqs[wids_unique]
        idfs = np.log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0

        # TF-IDF
        tfidfs = np.multiply(tfs, idfs)

        if len(tfidfs) <= k:
            o_sort = np.argsort(-tfidfs)
        else:
            o = np.argpartition(-tfidfs, k)[0:k]
            o_sort = o[np.argsort(-tfidfs[o])]


        # normalize score to integer between [0, 9]
        words2tfidf = {word2idx[wids2words[wid]]: max(0, min(round(tfidf), 9)) for wid, tfidf in zip(wids_unique[o_sort], tfidfs[o_sort])
                       if wids2words[wid] in word2idx and word2idx[wids2words[wid]] < vocab_size and wids2words[wid] not in stoplist}
        return words2tfidf
