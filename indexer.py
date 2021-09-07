# DO NOT MODIFY CLASS NAME
import math
import pickle
import os


class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        self.inverted_idx = {}
        self.config = config
        self.corpus_size = 0
        self.doc_length_dic = {}

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
                This function perform indexing process for a document object.
                Saved information is captures via two dictionaries ('inverted index' and 'posting')
                :param document: a document need to be indexed.
                :return: -
                """
        document_dictionary = document.term_doc_dictionary

        document_id = document.tweet_id
        doc_length = document.doc_length
        if doc_length > 0:
            self.doc_length_dic[document_id] = [doc_length]
        if document.max_tf>0:
            self.doc_length_dic[document_id].append(document.max_tf)
        if document.num_of_terms>0:
            self.doc_length_dic[document_id].append(document.num_of_terms)
        self.corpus_size += 1

        # Go over each term in the doc
        for term in document_dictionary:

            if term not in self.inverted_idx:
                self.inverted_idx[term] = [{document_id: document_dictionary[term]}, 1]
            else:
                self.inverted_idx[term][0][document_id] = document_dictionary[term]
                self.inverted_idx[term][1] += 1

    def check_entities(self, entities_dic):
        """
        This function add the entities to indexer
        :param entities_dic: dictionary, entities dictionary to add
        :return:
        """
        for entity in entities_dic:
            if entities_dic[entity][1] < 2:  # if entity was seen only one time in corpus
                continue
            self.inverted_idx[entity][1] += 1  # update indexer

            for tweet_id in entities_dic[entity][0]:
                self.update_change(tweet_id,
                                   entities_dic[entity][0][tweet_id])  # add document change to changes dctionary

            self.inverted_idx[entity][0].update(entities_dic[entity][0])  # update posting with entity

    def check_upper_letters(self, upper_letters_dic, low_letters_dic):
        """
        This function adds all upper words to indexer
        :param low_letters_dic: low letters dictionary
        :param upper_letters_dic: upper letters indexer
        :return:
        """
        for upper_word in upper_letters_dic:
            lower_word = upper_word.lower()
            if lower_word in low_letters_dic:  # check if this word was seen in lower letters
                if lower_word not in self.inverted_idx:
                    self.inverted_idx[lower_word] = [{}, upper_letters_dic[upper_word][1]]
                else:
                    self.inverted_idx[lower_word][1] += upper_letters_dic[upper_word][1]

                for tweet_id in upper_letters_dic[upper_word][0]:
                    self.update_change(tweet_id, upper_letters_dic[upper_word][0][
                        tweet_id])  # add document change to changes dctionary
                    self.inverted_idx[lower_word][0][tweet_id] = upper_letters_dic[upper_word][0][tweet_id]

            else:

                if upper_word not in self.inverted_idx:  # update indexer with upper word
                    self.inverted_idx[upper_word] = [{}, upper_letters_dic[upper_word][1]]
                else:
                    self.inverted_idx[upper_word][1] += upper_letters_dic[upper_word][1]

                for tweet in upper_letters_dic[upper_word][0]:
                    self.update_change(tweet, upper_letters_dic[upper_word][0][
                        tweet])  # add document change to changes dictionary
                    if tweet in self.inverted_idx[upper_word][0]:
                        self.inverted_idx[upper_word][0][tweet] += upper_letters_dic[upper_word][0][tweet]
                    else:
                        self.inverted_idx[upper_word][0][tweet] = upper_letters_dic[upper_word][0][tweet]

    def update_change(self, tweet_id, num_of_shows):
        """
        This function adds a change to changes dictionary in purpose of updating all the documents in the end
        :param tweet_id: int,tweet to update
        :param num_of_shows: int,num of shows of term to update
        :return:
        """

        if tweet_id not in self.doc_length_dic:
            self.doc_length_dic[tweet_id] = [num_of_shows,num_of_shows,1]

        else:
            self.doc_length_dic[tweet_id][0] += num_of_shows
            self.doc_length_dic[tweet_id][2] += 1
            if num_of_shows>self.doc_length_dic[tweet_id][1]:
                self.doc_length_dic[tweet_id][1] = num_of_shows

    def finish_indexing(self, entities_dic, low_letters_dic, upper_letters_dic):
        """
        This function add all the problematic terms to indexer and finish indexing
        :param entities_dic: dict,entities dictionary to add to indexer
        :return:
        """
        self.check_entities(entities_dic)  # add entities to files dictionary
        self.check_upper_letters(upper_letters_dic, low_letters_dic)  # add upper letters to indexet
        self.compute_nf_for_documents()  # compute nf for each document
        self.save_index('indexer')  # save indexer file to disk

    def compute_nf_for_documents(self):
        """
        This function computes the nf for each document
        :return:
        """
        docs_weights_Sum_dic = {}
        for term in self.inverted_idx:  # loop over all documents and compute nf
            for doc in self.inverted_idx[term][0]:
                number_of_shows_in_doc = self.inverted_idx[term][0][doc]
                max_tf = self.doc_length_dic[doc][1]
                num_of_terms = self.doc_length_dic[doc][2]
                doc_length = self.doc_length_dic[doc][0]
                if doc not in docs_weights_Sum_dic:
                    docs_weights_Sum_dic[doc] = pow(number_of_shows_in_doc /(0.95*doc_length + 0.05*num_of_terms), 2)
                else:
                    docs_weights_Sum_dic[doc] += pow(number_of_shows_in_doc / (0.95*doc_length + 0.05*num_of_terms ), 2)

        for term in self.inverted_idx:  # loop over all indexer and update for each term the posting
            idf = math.log10(self.corpus_size / self.inverted_idx[term][1])
            for document in self.inverted_idx[term][0]:
                number_of_shows_in_document = self.inverted_idx[term][0][document]
                doc_lentgh = self.doc_length_dic[document][0]
                max_tf = self.doc_length_dic[document][1]
                num_of_terms = self.doc_length_dic[document][2]
                tf_idf = (number_of_shows_in_document /(0.15*num_of_terms+0.85*max_tf) ) * idf
                doc_nf = 1 / math.sqrt(docs_weights_Sum_dic[document])
                self.inverted_idx[term][0][document] = [number_of_shows_in_document, tf_idf, doc_nf]
            self.inverted_idx[term].append(len(self.inverted_idx[term][0]))

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        file_path = os.path.join(self.config.savedFileMainFolder, fn)
        with open(file_path + ".pkl", 'rb') as f:
            indexer = pickle.load(f)
        return indexer

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        file_path = os.path.join(self.config.savedFileMainFolder, fn)
        with open(file_path + ".pkl", 'wb') as f:
            pickle.dump(self.inverted_idx, f)
