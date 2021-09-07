# you can change whatever you want in this module, just make sure it doesn't 
# break the searcher module
import math


class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_doc(relevant_doc, query, k=None):
        """
        This function computes the similarity for the documents with the query
        :param relevant_doc: dict,a dictionary containing for key a term and for value the whole posting file of the term
        :param query: list, a list of words in the query
        :param k: int,number of documents to return
        :return: tuple, in the first place the number of documents returned and in the second place all the relevant tweets sorted in descending order
        """
        query_terms_dict = {}
        sim = {}
        for term in query:  # compute every term occurnces in query
            if term not in query_terms_dict:
                query_terms_dict[term] = 1

        query_lentgh = 1 / (math.sqrt(len(query_terms_dict)))

        for term in query_terms_dict:  # loop over each term in doc list and compute for each document similarity
            if term in relevant_doc:
                for doc in relevant_doc[term]:
                    tf_idf = relevant_doc[term][doc][1]
                    if doc in sim:
                        sim[doc][0] += tf_idf

                    else:
                        sim[doc] = [tf_idf, relevant_doc[term][doc][2]]

        for doc in sim:
            sim[doc] = sim[doc][0] * query_lentgh * sim[doc][1]  # compute cos similarity for each doc

        results = sorted(sim.items(), key=lambda item: item[1], reverse=True)
        return Ranker.create_return_value(results, k)

    @staticmethod
    def create_return_value(results, k=None):
        """
        This method creates the return value as mentioned in rank_relevant_doc documentation
        :param results: list,a list of tuples of the relevant tweets,in the first place the tweet id and in the second place it's rank
        :param k:
        :return:
        """
        results_list = []
        if k is not None:
            if len(results)<k:
                for i in range(len(results)):
                    results_list.append(results[i][0])
            else:
                for i in range(k):
                    results_list.append(results[i][0])
            return len(results_list), results_list
        for tweet_result in results:
            results_list.append(tweet_result[0])
        return len(results_list), results_list
