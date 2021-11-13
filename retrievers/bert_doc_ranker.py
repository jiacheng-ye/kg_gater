"""
This example uses Approximate Nearest Neighbor Search (ANN) with FAISS (https://github.com/facebookresearch/faiss).
Searching a large corpus with Millions of embeddings can be time-consuming. To speed this up,
ANN can index the existent vectors. For a new query vector, this index can be used to find the nearest neighbors.
This nearest neighbor search is not perfect, i.e., it might not perfectly find all top-k nearest neighbors.
In this example, we use FAISS with an inverse flat index (IndexIVFFlat). It learns to partition the corpus embeddings
into different cluster (number is defined by n_clusters). At search time, the matching cluster for query is found and only vectors
in this cluster must be search for nearest neighbors.
This script will compare the result from ANN with exact nearest neighbor search and output a Recall@k value
as well as the missing results in the top-k hits list.
See the FAISS repository, how to install FAISS.
As dataset, we use the Quora Duplicate querys dataset, which contains about 500k querys (only 100k are used):
https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-query-Pairs.
As embeddings model, we use the SBERT model 'distilbert-multilingual-nli-stsb-quora-ranking',
that it aligned for 100 languages. I.e., you can type in a query in various languages and it will
return the closest querys in the corpus (querys in the corpus are mainly in English).
"""
from sentence_transformers import SentenceTransformer
import os
import pickle
import faiss
from faiss import normalize_L2
import numpy as np
from retrievers.utils import read_tokenized_src_file
from sklearn.feature_extraction.text import TfidfVectorizer
import string

stoplist = list(string.punctuation)


class SBERTDocRanker():
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, opt):
        self.opt = opt
        # model_name = 'bert-base-nli-mean-tokens'
        model_name = "allenai-specter"
        # model_name = "output/training_continue_training-allenai-specter-2021-05-09_07-17-03"
        self.model = SentenceTransformer(model_name)
        embed_cache_path = opt.save_data_dir + '/kp20k-embeddings-{}.pkl'.format(model_name.replace('/', '_'))
        # embed_cache_path = 'data/kp20k_sorted50/Full50_Dense_RefKP_RefDoc_RefGraph_CopyRef/kp20k-embeddings-{}.pkl'.format(model_name.replace('/', '_'))
        self.index, self.tfidf_vectorizer = self.build_index(embed_cache_path)

    def build_index(self, embed_cache_path):
        embedding_size = 768  # Size of embeddings
        # Defining our FAISS index
        # Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N)
        # - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
        # n_clusters = 5600  # N=500000
        n_clusters = 2000  # N=50000
        # n_clusters = 20  # N=50000

        # We use Inner Product (dot-product) as Index. We will normalize our vectors
        # to unit length, then is Inner Product equal to cosine similarity
        quantizer = faiss.IndexFlatIP(embedding_size)
        index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)
        # index = faiss.IndexFlatIP(embedding_size)

        # Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
        index.nprobe = 200
        # index.nprobe = 3

        # Check if embedding cache path exists
        if not os.path.exists(embed_cache_path):
            ref_docs = read_tokenized_src_file(self.opt.ref_doc_path)
            corpus_embeddings = self.model.encode(ref_docs, show_progress_bar=True, convert_to_numpy=True)

            ### Create the FAISS index
            print("Start creating FAISS index")
            # First, we need to normalize vectors to unit length
            normalize_L2(corpus_embeddings)
            # corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

            tfidf_vectorizer = TfidfVectorizer(tokenizer=str.split,
                                               vocabulary={w: i for w, i in self.opt.vocab["word2idx"].items()
                                                           if i < self.opt.vocab_size})
            tfidf_vectorizer.fit(ref_docs)

            id2word = {}
            for w, id in tfidf_vectorizer.vocabulary_.items():
                id2word[id] = w
            tfidf_vectorizer.id2word = id2word

            print("Store corpus_embeddings on disc")
            with open(embed_cache_path, "wb") as fOut:
                pickle.dump({'corpus_embeddings': corpus_embeddings,
                             'tfidf_vectorizer': tfidf_vectorizer}, fOut)
        else:
            print("Load pre-computed embeddings from disc")
            with open(embed_cache_path, "rb") as fIn:
                cache_data = pickle.load(fIn)
                corpus_embeddings = cache_data['corpus_embeddings']
                tfidf_vectorizer = cache_data['tfidf_vectorizer']

        # Then we train the index to find a suitable clustering
        index.train(corpus_embeddings)

        # Finally we add all embeddings to the index
        index.add(corpus_embeddings)

        return index, tfidf_vectorizer
        
    def batch_closest_docs(self, queries, k=1):
        query_embedding = self.model.encode(queries, show_progress_bar=True, convert_to_numpy=True)

        # FAISS works with inner product (dot product). When we normalize vectors to unit length,
        # inner product is equal to cosine similarity
        normalize_L2(query_embedding)

        # Search in FAISS. It returns a matrix with distances and corpus ids.
        distances, corpus_ids = self.index.search(query_embedding, k)

        # normalize score to integer between [0, 9]
        distances = np.round(distances * 9)
        distances[distances < 0] = 0
        return corpus_ids, distances

    def batch_words_tfidf(self, queries, k=20, word2idx=None):
        batch_tfidf = self.tfidf_vectorizer.transform(queries)

        # batch_sorted_idx, batch_tfidf = row_topk_csr(batch_tfidf.data, batch_tfidf.indices, batch_tfidf.indptr, k)
        batch_tfidf, batch_sorted_idx = top_k(batch_tfidf, k)

        # normalize score to integer between [0, 9]
        words2tfidf = [{word2idx[self.tfidf_vectorizer.id2word[id]]: np.round(tfidf * 9)
                        for id, tfidf in zip(topic_words_id, words_tfidf)
                        if id != -1 and self.tfidf_vectorizer.id2word[id] not in stoplist}
                       for topic_words_id, words_tfidf in zip(batch_sorted_idx, batch_tfidf)]
        return words2tfidf


def _top_k(d, r, k):
    return zip(*sorted(zip(d, r), reverse=True)[:k])


def top_k(m, k):
    """
    Keep only the top k elements of each row in a csr_matrix
    """
    ml = m.tolil()
    ms = [_top_k(d, r, k) for d, r in zip(ml.data, ml.rows)]
    return zip(*ms)

