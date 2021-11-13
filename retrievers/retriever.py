from .utils import read_src_and_trg_files, read_tokenized_src_file, get_node_edge
import random


class Retriever():
    def __init__(self, opt):
        self.opt = opt
        if self.opt.ref_kp_path != "":
            self.ref_docs = read_src_and_trg_files(opt.ref_doc_path, opt.ref_kp_path, opt.ref_doc, opt.ref_kp)
        else:
            self.ref_docs = read_tokenized_src_file(opt.ref_doc_path)

        if opt.dense_retrieve:
            from retrievers.bert_doc_ranker import SBERTDocRanker
            self.ranker = SBERTDocRanker(opt)
        else:
            from retrievers.tfidf_doc_ranker import TfidfDocRanker
            self.ranker = TfidfDocRanker(opt.hash_path)


    def maybe_retrieving_building_graph(self, source, word2idx, vocab_size, is_train=False):
        if self.opt.n_ref_docs > 0:
            if self.opt.random_search:
                random.seed(1235)
                ref_doc_texts = random.sample(self.ref_docs, self.opt.n_ref_docs)
                ref_doc_scores = [random.randint(0, 9) for _ in range(self.opt.n_ref_docs)]  # tmp value
            else:
                if is_train:  # ignore the same doc
                    ref_docids, ref_doc_scores = self.ranker.closest_docs(source, k=self.opt.n_ref_docs+1)
                    ref_doc_scores = ref_doc_scores[1:]
                    ref_docids = ref_docids[1:]
                else:
                    ref_docids, ref_doc_scores = self.ranker.closest_docs(source, k=self.opt.n_ref_docs)

                ref_doc_texts = [self.ref_docs[ref_docid] for ref_docid in ref_docids]
            ref_doc_texts_tokenized = [d.split() for d in ref_doc_texts]
        else:
            ref_doc_texts = []
            ref_doc_texts_tokenized = []

        if self.opt.n_topic_words > 0:
            cur_tfidfs = self.ranker.words_tfidf(source, self.opt.n_topic_words, word2idx, vocab_size)
            ref_tfidfs = [self.ranker.words_tfidf(d, self.opt.n_topic_words, word2idx, vocab_size) for d in ref_doc_texts]

            # ref_doc_scores = list(range(len(ref_doc_texts_tokenized)))
            graph_utils = get_node_edge([source.split()] + ref_doc_texts_tokenized, [cur_tfidfs] + ref_tfidfs, ref_doc_scores)
        else:
            graph_utils = None

        return ref_doc_texts_tokenized, graph_utils

    def batch_maybe_retrieving_building_graph(self, source, word2idx, vocab_size, is_train=False):
        if self.opt.n_ref_docs > 0:
            if self.opt.random_search:
                batch_ref_doc_texts = random.sample(self.ref_docs, self.opt.n_ref_docs)
                batch_ref_doc_scores = [1] * self.opt.n_ref_docs  # tmp value
            else:
                if is_train:  # ignore the same doc
                    batch_ref_docids, batch_ref_doc_scores = self.ranker.batch_closest_docs(source, k=self.opt.n_ref_docs+1)
                    batch_ref_doc_scores = batch_ref_doc_scores[:, 1:]
                    batch_ref_docids = batch_ref_docids[:, 1:]
                else:
                    batch_ref_docids, batch_ref_doc_scores = self.ranker.batch_closest_docs(source, k=self.opt.n_ref_docs)

                batch_ref_doc_texts = [[self.ref_docs[ref_docid] for ref_docid in ref_docids] for ref_docids in batch_ref_docids]
            batch_ref_doc_texts_tokenized = [[d.split() for d in ref_doc_texts] for ref_doc_texts in batch_ref_doc_texts]
        else:
            batch_ref_doc_texts = None
            batch_ref_doc_texts_tokenized = None

        if self.opt.n_topic_words > 0:
            batch_cur_tfidfs = self.ranker.batch_words_tfidf(source, self.opt.n_topic_words, word2idx)
            batch_ref_tfidfs = [self.ranker.batch_words_tfidf(ref_doc_texts, self.opt.n_topic_words, word2idx)
                                for ref_doc_texts in batch_ref_doc_texts]

            # ref_doc_scores = list(range(len(ref_doc_texts_tokenized)))
            graph_utils = [get_node_edge([cur_doc_text.split()] + ref_doc_texts_tokenized, [cur_tfidfs] + ref_tfidfs, ref_doc_scores)
                           for cur_doc_text, ref_doc_texts_tokenized, cur_tfidfs, ref_tfidfs, ref_doc_scores in zip(source,
                                                                                                                    batch_ref_doc_texts_tokenized,
                                                                                                                    batch_cur_tfidfs,
                                                                                                                    batch_ref_tfidfs,
                                                                                                                    batch_ref_doc_scores)]
        else:
            graph_utils = None

        return batch_ref_doc_texts_tokenized, graph_utils
