from nltk.corpus import wordnet
from spellchecker import SpellChecker
from ranker import Ranker
import utils


# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None):
        self._parser = parser
        self._indexer = indexer
        self._ranker = Ranker()
        self._model = model
        self.spell_checker = SpellChecker()

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """
        #query_as_list = self._parser.parse_sentence(query)
        self._parser.entities_dic = {}
        self._parser.low_letters_dic = {}
        self._parser.capital_letters_dic = {}
        query_as_list = query.split()
        query_tokens = []
        query_tokens.extend(self._parser.parse_sentence(query))  # parse query
        for term in query_as_list:
            if 'http' in term or 'www' in term:
                query_tokens.extend(self._parser.parse_url(term))  # parse url if in query

        low_letters_dic = self._parser.low_letters_dic
        capital_letters_dic = self._parser.capital_letters_dic
        entities_dic = self._parser.entities_dic
        self.add_missing_tokens_to_list(low_letters_dic, capital_letters_dic, entities_dic, query_tokens)
        relevant_docs, corrected_query = self._relevant_docs_from_posting(query_tokens)  # retrieve relevant docs for query terms
        if len(relevant_docs) == 0:  # check if has documents to retrieve
            return []

        ranked_docs = self._ranker.rank_relevant_doc(relevant_docs, corrected_query,k)  # rank the docs
        return ranked_docs

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _relevant_docs_from_posting(self, query_as_list):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param query_as_list: parsed query tokens
        :return: dictionary of relevant documents mapping doc_id to document frequency.
        """
        relevant_docs = {}
        corrected_query = []
        terms_dict = {}
        for word in query_as_list:  # make dictionary with unique terms
            if word not in terms_dict:
                terms_dict[word] = 1

        for term in terms_dict:  # loop over all the terms and order them by the file they're in
            if term[0].isalpha() and term.isalpha():  # check if term is a single word

                lst = self.correct_input(term)  # correct spelling
                if len(lst) > 0:
                    # synonyms = self.check_synonyms(lst[0]) # get synonyms of term
                    # for synonym in synonyms: # append each synonyms to corrected query
                    #     corrected_query.append(synonym)
                    #     relevant_docs[synonym] = self._indexer[synonym][0]

                    for word in lst:  # update posting dictionary with terms to extract
                        relevant_docs[word] = self._indexer[word][0]
                    corrected_query.append(lst[0])  # append word to corrected query

            else:
                if term in self._indexer:
                    relevant_docs[term] = self._indexer[term][0]  # update posting dictionary with terms to extract
                    corrected_query.append(term)  # append word to corrected query

        return relevant_docs, corrected_query


    def add_missing_tokens_to_list(self,low_letters_dic, capital_letters_dic, entities_dic, tokens_list):
            """
            This function adds upper letters and entities to query
            :param low_letters_dic: dictionary,low letters dictionary
            :param capital_letters_dic: dictionary,upper letters dictionary
            :param entities_dic: dictionary,entities dictionary
            :param tokens_list: list,list of tokens to add missing tokens to it
            :return:
            """

            for capital_word in capital_letters_dic:
                low_word = capital_word.lower()
                if low_word not in low_letters_dic:
                    tokens_list.append(capital_word)

            for entity in entities_dic:
                tokens_list.append(entity)


    def correct_input(self, token):
        """
        This function correct spelling errors
        :param token:string,word to check
        :return:string,corrected token
        """
        lst = self.check_word(token)  # check if the original word in indexer
        if len(lst) > 0:
            return lst
        corrected_word = self.spell_checker.correction(token)  # correct word
        if corrected_word.lower() != token.lower():
            lst = self.check_word(corrected_word)  # check if correction candidate in indexer
            if len(lst) > 0:
                return lst
        candidates = self.spell_checker.candidates(token)
        for candidate in candidates:  # loop over all candidates and checks if one of them in indexer
            lst = self.check_word(candidate)
            if len(lst) > 0:
                return lst
        edit_distance_1 = self.spell_checker.edit_distance_1(token)
        for elem in edit_distance_1:  # loop over all levinstein distance 1 and check if any of them in indexer
            lst = self.check_word(elem)
            if len(lst) > 0:
                return lst

        return []

    def check_word(self, word):
        """
        This function checks if lower word or upper word in indexer
        :param word: string,word to check
        :return:
        """
        lower_word = word.lower()
        upper_word = word.upper()
        if lower_word in self._indexer:
            return [lower_word]
        if upper_word in self._indexer:
            return [upper_word]
        return []

    def check_synonyms(self, word):
        """
        This function gets the synonyms that appear in the corpus of a given word
        :param word: str,the word to get the synonyms of
        :return: list, a list of the synonyms
        """
        synonyms_to_add_to_query = []
        lower_word = word.lower()
        for syn in wordnet.synsets(lower_word):
            for lemma in syn.lemmas():
                term = lemma.name().lower()
                if term != lower_word:
                    if '_' in term or '-' in term: # if synonym is more than a single word
                        continue
                    else:
                        word_in_corpus = self.check_word(term) # check if synonym in corpus
                        if len(word_in_corpus) > 0 and word_in_corpus[0] not in synonyms_to_add_to_query:
                            synonyms_to_add_to_query.append(word_in_corpus[0])

        return synonyms_to_add_to_query


