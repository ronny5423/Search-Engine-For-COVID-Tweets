import pandas as pd
import os
from configuration import ConfigClass
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None):
        self._config = config
        self._parser = Parse()
        self._indexer = Indexer(config)
        self._model = None

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def build_index_from_parquet(self, fn):
        """
        Reads parquet file and passes it to the parser, then indexer.
        Input:
            fn - path to parquet file
        Output:
            No output, just modifies the internal _indexer object.
        """
        df = pd.read_parquet(fn, engine="pyarrow")
        documents_list = df.values.tolist()
        # Iterate over every document in the file
        number_of_documents = 0
        for idx, document in enumerate(documents_list):
            # parse the document
            parsed_document = self._parser.parse_doc(document)
            number_of_documents += 1
            # index the document data
            self._indexer.add_new_doc(parsed_document)
        low_letters_dic = self._parser.low_letters_dic
        upper_letters_dic = self._parser.capital_letters_dic
        entities_dic = self._parser.entities_dic
        self._indexer.finish_indexing(entities_dic, low_letters_dic, upper_letters_dic)  # finish indexing

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        self._indexer.load_index(fn)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_precomputed_model(self, model_dir=None):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and 
        assign to self._model, which is passed on to the searcher at query time.
        """
        pass

        # DO NOT MODIFY THIS SIGNATURE
        # You can change the internal implementation as you see fit.

    def search(self, query):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """
        searcher = Searcher(self._parser, self._indexer.inverted_idx, model=self._model)
        return searcher.search(query)


def main():
    queries_path = os.getcwd() + "/data/queries_train.tsv"
    data_frame = pd.read_csv(queries_path, sep="\t")
    config = ConfigClass()
    search_engine = SearchEngine(config)
    search_engine.build_index_from_parquet(config.corpusPath)
    for idx, row in data_frame.iterrows():
        query = row[1]
        results = search_engine.search(query)
        print(str(query))
        print(results[1])
