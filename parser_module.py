import os
import pickle
import re
import json
from urllib.parse import urlparse
from nltk.corpus import stopwords
from document import Document
from nltk.stem import PorterStemmer

months = {'january': '01', 'february': '02', 'march': '03', 'april': '04',
          'may': '05', 'june': '06', 'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11',
          'december': '12'}

months_3_letters_dic = {'jan': 'january', 'feb': 'february', 'mar': 'march', 'apr': 'april', 'may': 'may',
                        'jun': 'june', 'jul': 'july', 'aug': 'august',
                        'sep': 'september', 'oct': 'october', 'nov': 'november', 'dec': 'december'}


def can_it_be_numeric(num):
    """
    This function checks if a token can be numeric
    :param num: the token to check
    :return: true if it can be or false otherwise
    """
    if '.' in num:  # check if . in the token and not in the first place
        if num[-1] != "." and num[0] != "." and num.replace('.', '').isnumeric():
            return True
        return False
    if num[-1] != "," and num[0] != "," and num.replace(',', '').isnumeric():  # check the same for ,
        return True
    return False


def split_string(string_to_split, delimiters):
    """
    This function gets a string and list of delimiters and split the string using these delimiters
    :param string_to_split: the string t split
    :param delimiters: delimiters to split by
    :return: list of splited tokens
    """
    splited_string_list = re.split(delimiters, string_to_split)
    for item in splited_string_list:
        if item in delimiters:
            splited_string_list.remove(item)
    return splited_string_list


def handles_numbers(num):
    """
    This function convers a number token by the rules we were told to do
    :param num: the num to fix
    :return: fixed string
    """
    num = num.replace(',', "")

    if 1000 > float(num):
        if '.' in num:
            return str(round(float(num), 3))
        else:
            return num
    num = float(num)
    str_to_return = str(num)
    if 1000 <= num < pow(10, 6):
        str_to_return = str(round(num / 1000, 3)) + "K"
    elif pow(10, 6) <= num < pow(10, 9):
        str_to_return = str(round(num / pow(10, 6), 3)) + "M"
    elif pow(10, 9) <= num < pow(10, 12):
        str_to_return = str(round(num / pow(10, 9), 3)) + "B"
    if str_to_return[-2] == "0":
        return str_to_return[:-3] + str_to_return[-1]
    else:
        return str_to_return


def remove_chars_from_string(text, delimiters):
    """
    This function gets a string and delimiters and remove the delimiters from the string
    :param text: string to remove from
    :param delimiters:list, delimiters to remove
    :return: string withous those delimiters
    """
    text = re.sub(delimiters, '', text)
    return text


def check_if_month(word):
    """
    This function checks if a string is a month
    :param word: string, word to check
    :return: tuple that contains the key of the dictionary and the month string in lower letters or None if not a month
    """
    if not word.islower():
        word = word.lower()
    if word in months_3_letters_dic:
        word = months_3_letters_dic[word]
    if word in months:
        return word, months[word]
    return None


def delete_emojis(text):
    """
    This function deletes from a string every char that not in ascii table
    :param text: string,the string to remove from
    :return: string without non ascii chars
    """
    return text.encode("ascii", "ignore").decode()


def check_term(term):
    """
    This function delete from the begining of the string every char that is not alphabetic,number,# or @
    :param term: string,the string to remove from
    :return: string that the first char of it's is one of the above mentioned earlier
    """
    for i in range(len(term)):
        if term[i].isalpha() or term[i].isnumeric() or term[i] in ['#', '@']:
            return term[i:]

    return None


def init_merged_data_dic(file_path):
    """
    This function load a dctionary from a file
    :param file_path: string,the path of the file
    :return: dictionary from file
    """
    pickle_file = open(file_path, "rb")
    merged_data_dic = pickle.load(pickle_file)
    pickle_file.close()
    return merged_data_dic


def convert_number(token, index):
    """
    This function deal with string tokens that include number and b,m or k in the end
    :param token: string,the token to fix
    :param index: index of the token in the list
    :return: string, fixed token
    """
    if token[index] in ['b', 'B']:
        return handles_numbers(str(float(token[:index]) * 1000000000))

    elif token[index] in ['M', 'm']:
        return handles_numbers(str(float(token[:index]) * 1000000))

    elif token[index] in ['k', 'K']:
        return handles_numbers(str(float(token[:index]) * 1000))


class Parse:

    def __init__(self, with_stemmer=False):
        self.stop_words = stopwords.words('english')
        self.add_stop_words()
        self.tokens = []
        self.num_tuple = (-10, "-10")
        self.date_tuple = [-10, "-10", False]
        self.stemmer = None
        self.use_stemmer = with_stemmer
        if with_stemmer:
            self.stemmer = PorterStemmer()
        self.capital_letters_dic = {}
        self.low_letters_dic = {}
        self.entities_dic = {}
        # self.tweets_counter = 0
        # self.files_counter = 0
        # self.upper_letters_path = os.path.join(output_path,'upper')

    def add_stop_words(self):
        with open("stopWords.txt",'r') as f:
            lines = f.readlines()
        for line in lines:
            self.stop_words.append(line)

    def parse_sentence(self, text, tweet_id=-1):
        """
        This function tokenize, remove stop words and apply lower case for every word within the text
        :param tweet_id:int,tweet id we're working on
        :param text:string,the text to parse
        :return: list,list of tokens of the text
        """
        text = delete_emojis(text)
        text_tokens = text.split()
        if len(text_tokens) == 0:
            return []
        index = 0
        self.date_tuple = [-10, "-10", False]
        self.num_tuple = (-10, "-1")
        self.tokens = []
        names = [[], -10]

        if text_tokens[0] == 'RT':
            text_tokens.pop(0)
            if len(text_tokens) > 0 and text_tokens[0][0] == '@':
                self.tokens.append(text_tokens[0][:-1])
                text_tokens.pop(0)

        for token in text_tokens:
            try:
                if len(names[0]) > 1 and names[1] != index:  # check if we can add entity that in the list to dictionary
                    self.add_entity_to_entities_dic(names, tweet_id)
                    names = [[], -10]

                token = check_term(token)
                if token is None:
                    index += 1
                    continue
                token = remove_chars_from_string(token, '[`\+=?!"|()[]:<>~&*^;{}\.\.\.]')

                if len(token) == 0:
                    index += 1
                    continue

                if len(token) == 1:
                    if token.isnumeric():
                        self.num_tuple = (index, token, token)
                        self.tokens.append(token)
                    index += 1
                    continue

                elif 'www' in token or 'http' in token:  # skip url
                    index += 1
                    continue

                elif token[0] == '#':
                    self.parse_hashtag(token)

                elif token[0] == '@':
                    self.tokens.append(token)

                elif '$' in token and self.parse_dollar(token):  # parse dollar
                    index += 1
                    continue

                # elif '-' in token: # parse hyphen
                #     token = remove_chars_from_string(token, '[.,$]')
                #     self.parse_hyphen(token, names, index, tweet_id)

                elif token[0].isnumeric():  # parse numbers
                    token = remove_chars_from_string(token, '[$]')
                    self.parse_numbers(token, index)

                elif self.num_tuple[0] + 1 == index and self.check_units(token, index):  # parse units
                    index += 1
                    continue

                elif token[0].isalpha() and token.isalpha():  # parse words
                    self.check_words(token, index, tweet_id, names, text_tokens)

                else:  # if none from the whole conditions
                    #print(token)
                    token = remove_chars_from_string(token, '[.,$/]')
                    self.tokens.append(token)

                index += 1
            except:
                index += 1

        if len(names[0]) > 1:
            self.add_entity_to_entities_dic(names, tweet_id)

        return self.tokens

    def add_entity_to_entities_dic(self, names, tweet_id):
        """
        This function adds entity to entities dic if needed
        :param names: list,entitity's tokens
        :param tweet_id: int,tweet id
        :return:
        """
        entity = ' '.join(names[0])
        entity = entity.upper()

        if entity in self.entities_dic:
            if tweet_id in self.entities_dic[entity][0]:  # check if entity was seen again in the same tweet
                self.entities_dic[entity][0][tweet_id] += 1
            else:
                if self.entities_dic[entity][1] == 1:  # if entity was seen once already
                    self.entities_dic[entity][1] += 1
                self.tokens.append(entity)  # append entity to tokens


        else:  # enter entity to entity's dictionary
            self.entities_dic[entity] = [{tweet_id: 1}, 1]
            # self.entities_dic[entity].extend(names[0])

        for token in names[0]:  # append entity's tokens to tokens list
            num_of_shows = list(self.entities_dic[entity][0].values())
            to_add = self.check_if_to_add_word(token, tweet_id, num_of_shows[0])
            if to_add is not None:
                self.tokens.append(to_add)

    # def parse_hyphen(self, token, names, index, tweet_id):
    #     """
    #     This function parse expressions with hyphen
    #     :param token: string,the token to parse
    #     :param names: list,list of collected entity
    #     :param index: int, index of the token
    #     :param tweet_id: int,current tweet id
    #     :return:
    #     """
    #     self.tokens.append(token)
    #     parsed_token = token.split('-')
    #     for i in range(len(parsed_token)):
    #         if parsed_token[i]:
    #             if parsed_token[i][0].isupper() and (
    #                     (len(names[0]) == 0 and index + 1 < len(parsed_token) and  # check if part of entity
    #                      parsed_token[i + 1][0].isupper()) or (
    #                             names[1] + 1 == index)):
    #                 names[0].append(parsed_token[i])
    #                 names[1] = index
    #             else:
    #                 if parsed_token[i].isalpha():
    #                     token_to_add_to_tokens = self.check_if_to_add_word(parsed_token[i], tweet_id)
    #                     if token_to_add_to_tokens is not None:
    #                         self.tokens.append(token_to_add_to_tokens)
    #                 else:
    #                     self.tokens.append(parsed_token[i])

    def parse_doc(self, doc_as_list):
        """
        This function takes a tweet document as list and break it into different fields
        :param doc_as_list: list re-preseting the tweet.
        :return: Document object with corresponding fields.
        """
        tweet_id = doc_as_list[0]
        tweet_date = doc_as_list[1]
        full_text = doc_as_list[2]
        url = doc_as_list[3]
        retweet_text = doc_as_list[5]
        retweet_url = doc_as_list[6]
        quote_text = doc_as_list[8]
        quote_url = doc_as_list[9]
        term_dict = {}
        url_dict = {}
        tokenize_text = []

        if retweet_text is not None and retweet_text != '{}' and 'http' not in retweet_text:  # decide if parse retweet text or full text
            lst = self.parse_sentence(retweet_text, tweet_id)
            if len(lst) > 0:
                tokenize_text.extend(lst)
            # strudel = full_text.find('@')
            # dots = full_text.find(':')
            # if strudel != -1 and dots != -1:  # extract retweet user from full text
            #     tokenize_text.append(full_text[strudel:dots])

        else:
            lst = self.parse_sentence(full_text, tweet_id)
            if len(lst) > 0:
                tokenize_text.extend(lst)

        if quote_text is not None:  # parse quote text
            lst = self.parse_sentence(quote_text, tweet_id)
            if len(lst) > 0:
                tokenize_text.extend(lst)
        # try:
        #     if url is not None and url != '{}':  # add url to urls dictionary
        #         url_dict.update(json.loads(url))
        #     if quote_url is not None and quote_url != '{}':  # add quote url to urls dictionary
        #         url_dict.update(json.loads(quote_url))
        #
        #     if retweet_url is not None and retweet_url != '{}' and (url is None or url == '{}'):
        #         url_dict.update(json.loads(retweet_url))
        #
        #     if bool(url_dict):
        #         for short_url in url_dict:  # parse each url
        #             tokenize_text.extend(self.parse_url(url_dict[short_url]))
        # except:
        #     i = 5

        max_len = 0
        for term in tokenize_text:
            term = check_term(term)
            if term is None:
                continue
            if len(term) == 0:
                continue
            if term not in term_dict.keys():
                term_dict[term] = 1
            else:
                term_dict[term] += 1
        len_of_dic = 0
        for key in term_dict:
            if term_dict[key] > max_len:
                max_len = term_dict[key]
            len_of_dic += term_dict[key]

        document = Document(tweet_id, tweet_date, full_text, url, retweet_text, retweet_url, quote_text,
                            quote_url, term_dict, len_of_dic, max_len)

        # if self.tweets_counter == 500000:
        #     self.load_to_file()  # load to file capital letters dic
        # else:
        #     self.tweets_counter += 1

        return document

    def add_sign(self, sign, num, tok):
        """
        This function adds a sign to token
        :param sign: string,sign to add
        :param num: int,index of the token
        :param tok: string,the token to add sign to
        :return:
        """
        if self.num_tuple[0] is num - 1:
            self.tokens = self.tokens[:-1]
            self.tokens.append(str(self.num_tuple[1]) + sign)
            # self.num_tuple = (num, self.tokens[-1], "0")
        else:
            self.tokens.append(tok)

    def check_date_text(self, text, delimiter):
        """
        This function checks if a token is a date
        :param text: string,text to check
        :param delimiter: string,delimiter of the token,/ or -
        :return:
        """
        if not text.replace(delimiter, '').isnumeric():  # if the token not numeric without the delimiter
            return False

        parsed_date = text.split(delimiter)
        if len(parsed_date) == 3 or len(parsed_date) == 2:
            for token in parsed_date:
                self.tokens.append(token)
            self.tokens.append(text)
            return True
        return False

    def check_units(self, token, index):
        """
        This function checks if a current token is unit from the list below and append it to a number
        :param token: string,token to check
        :param index: int,index of the token
        :return: return true if could be appended or false otherwise
        """
        if token[0].isalpha() and token.isalpha():
            text = token.lower()
        else:
            return False

        lower_text = text.lower()
        if lower_text in ["percent", "percentage"]:
            self.add_sign("%", index, token)
        elif lower_text in ["dollar", "dollars"]:
            self.add_sign("$", index, token)
        elif lower_text in ["thousand", "thousands"]:
            self.add_sign("K", index, token)
        elif lower_text in ["million", "millions"]:
            self.add_sign("M", index, token)
        elif lower_text in ["billion", "billions"]:
            self.add_sign("B", index, token)
        else:
            return False
        return True

    def check_words(self, token, index, tweet_id, names, tokens_list):
        """
        This function parse words
        :param token: string,token to parse
        :param index: int,index of the token
        :param tweet_id: int,current tweet id
        :param names: list,list of entity names that were collected so far
        :param tokens_list: list,list of text tokens we parse
        :return:
        """
        splitted_text = split_string(token, '[.,/]')  # split string by these delimiters
        for i in range(len(splitted_text)):
            if splitted_text[i] or len(splitted_text[i]) > 1:
                if splitted_text[i].lower() in self.stop_words and names[1] + 1 != index:  # check if token is stop word
                    continue

                month = check_if_month(splitted_text[i])  # check if the word is a month
                if month is not None:  # if yes
                    text = self.check_if_to_add_word(splitted_text[i], tweet_id)
                    if text is not None:
                        self.tokens.append(text)
                    if self.num_tuple[0] + 1 == index and self.num_tuple[2] != '0':  # save the month to tuple
                        number = int(self.num_tuple[1])
                        if 1 <= number <= 31:  # check if the number we seen earlier is day
                            self.tokens.append(str(number) + "/" + month[1])
                        else: # year
                            self.tokens.append(month[1] + "/" + str(number))
                    else:
                        self.date_tuple = [index, month[0]]

                elif splitted_text[i][0].isupper() and not (i + 1 < len(splitted_text)) and (
                        # check if token is part of entity
                        (len(names[0]) == 0 and index + 1 < len(tokens_list) and tokens_list[index + 1][
                            0].isupper()) or (
                                names[1] + 1 == index)):
                    names[0].append(token)
                    names[1] = index

                else:  # not a month or entity
                    text = self.check_if_to_add_word(splitted_text[i], tweet_id)
                    if text is not None:
                        self.tokens.append(text)

    def parse_numbers(self, token, index):
        """
        This function parse number tokens
        :param token: string,number token
        :param index: int,index of current token
        :return:
        """
        if '/' in token and self.num_tuple[0] + 1 != index and self.check_date_text(token,
                                                                                    '/'):  # check if a date seperated by /
            return

        elif '-' in token and self.check_date_text(token, '-'):  # check if a date seperated by -
            return

        elif token[-1].isdigit() and can_it_be_numeric(token):
            # check if the number is coming after a month
            if self.date_tuple[0] + 1 == index and '.' not in token and ',' not in token:
                # check if the previous token was a date and if the current number doesn't contain . or ,
                number = int(token)
                if 1 <= number <= 31: # if day
                    self.tokens.append(token + '/' + months[self.date_tuple[1]])
                else: # year
                    self.tokens.append(months[self.date_tuple[1]] + '/' + token)
            else:  # regular number
                number = handles_numbers(token)
                self.num_tuple = (index, number, token.replace(',', ''))
                self.tokens.append(token)


        elif '/' in token and token.replace("/", "", 1).isnumeric():  # check if a fraction
            parsed_token = token.split("/")
            if parsed_token[0] != "" and parsed_token[1] != "":
                token = float(parsed_token[0]) / float(parsed_token[1])
                if self.num_tuple[0] + 1 == index:
                    token = float(self.num_tuple[1]) + token
                    self.tokens.pop(len(self.tokens) - 1)
                token = str(token)
                if token[-2:] == ".0":
                    token = token[:-2]

                token = handles_numbers(token)
                self.num_tuple = (index, token, '0')
                self.tokens.append(token)

        elif token[-2].isdigit() and can_it_be_numeric(
                token[:-1]):  # check if it's a number and after that there is % or unit
            token = token.replace(',', '')
            if "%" == token[-1]:
                self.tokens.append(handles_numbers(token[:-1]) + token[-1])
            elif token[-1].lower() in ['m', 'k', 'b']:
                self.tokens.append(convert_number(token, -1))
            else:
                self.tokens.append(token)

        elif can_it_be_numeric(token[:-2]):  # check if it's a number and after that there is % and unit
            if "%" == token[-1]:
                token = token.replace(',', '')
                char = token[-1]
                token = convert_number(token, -2)
                self.tokens.append(token + char)
        else:
            self.tokens.append(token)
        return True

    def parse_hashtag(self, token):
        """
        This function parse hashtag token
        :param token: string,token to parse
        :return: True if succseded to parse or false otherwise
        """
        token_without_hashtag = token[1:]
        if len(token_without_hashtag) < 2:
            return True
        #self.tokens.append(token.lower())

        if "_" in token_without_hashtag:  # check if - in hashtag
            lst = token_without_hashtag.split("_")
            for item in lst:
                item = remove_chars_from_string(item, '[.,$/]')
                if item.isalpha():
                    self.tokens.append(self.stem_word(item).lower())
                else:
                    self.tokens.append(item)
        elif token_without_hashtag[0].isupper() and token_without_hashtag.isupper():
            self.tokens.append(self.stem_word(token_without_hashtag).lower())
        else:  # split by capital letters
            lst = re.sub(r"([A-Z])", r" \1", token_without_hashtag).split()
            string = ""
            for item in lst:
                item = remove_chars_from_string(item, '[.,$/]')
                if len(item) == 1 and item.isupper():# check if each token is letter, COVID will be splitted to C O V I D and the letters alone mean nothing
                    string += item
                else: # a word
                    if item.isalpha():
                        self.tokens.append(self.stem_word(item).lower())
                    else:
                        self.tokens.append(item)
            if string: # if all the word was in capital letters
                if string.isalpha():
                    self.tokens.append(self.stem_word(string).lower())
                else:
                    self.tokens.append(string)
        return True

    def check_if_to_add_word(self, text, tweet_id, num_of_shows=1):
        """
        This function returns the current form of a word,lower or upper,if upper the function returns None
        :param text: string,the word to check
        :param tweet_id: int,current tweet id
        :param num_of_shows: int,num of shows in current occurance of the token
        :return:
        """
        text = self.stem_word(text) # check if to stem
        if text[0].islower() and text.islower():  # check if text was seen in lower
            if text not in self.low_letters_dic:
                self.low_letters_dic[text] = 1
            upper_word = text.upper()

            # check if the same word appeared previously in the same tweet in upper letters
            if upper_word in self.capital_letters_dic and tweet_id in self.capital_letters_dic[upper_word][0]:
                self.capital_letters_dic[upper_word][0][tweet_id] += num_of_shows
                return None
            return text

        upper_word = text.upper()
        if upper_word in self.capital_letters_dic:  # update upper word shows
            if tweet_id in self.capital_letters_dic[upper_word][0]:
                self.capital_letters_dic[upper_word][0][tweet_id] += num_of_shows
                return None
            self.capital_letters_dic[upper_word][0][tweet_id] = num_of_shows
            self.capital_letters_dic[upper_word][1] += 1
            return None

        self.capital_letters_dic[upper_word] = [{tweet_id: num_of_shows},1]  # if not seen,add upper word to dictionary
        return None

    def stem_word(self, word):
        """
        This function using the stemmer for word
        :param word: string,word to stem
        :return: stemmed word or the word itself
        """
        if self.use_stemmer:
            stemmed_word = self.stemmer.stem(word)  # stem word
            if word.islower():
                return stemmed_word
            return stemmed_word.upper()
        return word

    def parse_dollar(self, token):
        """
        This function parse tokens with $
        :param token: string,token with $
        :return: True if succeeded or false otherwise
        """
        if token[-1] == '$' and can_it_be_numeric(token[:-1]):
            number = handles_numbers(token[:-1])
            self.tokens.append(number + '$')
        elif token[-1] == '$' and can_it_be_numeric(token[:-2]):
            number = convert_number(token[:-1], -2)
            self.tokens.append(number + '$')
        else:
            return False
        return True

    # def load_to_file(self):
    #     """
    #     This function writes to file capital leeters dictionary
    #     :return:
    #     """
    #     for file_name in self.capital_letters_dic:
    #         file_path = os.path.join(self.upper_letters_path,file_name+str(self.files_counter))
    #         with open(file_path, 'wb') as f:
    #             pickle.dump(self.capital_letters_dic[file_name], f)
    #         self.capital_letters_dic[file_name] = {}
    #
    #     self.files_counter += 1
    #     self.tweets_counter = 0
    #
    # def merge_files(self):
    #     """
    #     This function merge all upper letters file into 27 files
    #     :return:
    #     """
    #     contents_dir = os.listdir(self.upper_letters_path)
    #     file_name_to_be_merged = ""
    #     merged_data_dic = {}
    #     for file in contents_dir:
    #         file_path = os.path.join(self.upper_letters_path, file)
    #         head, tail = os.path.split(file_path)
    #         if not file_name_to_be_merged:  # if first file of letter
    #             file_name_to_be_merged = tail[0]
    #             merged_data_dic = init_merged_data_dic(file_path)
    #
    #         elif tail[0] == file_name_to_be_merged[0]:  # if file in the same letter as we merge
    #             pickle_file = open(file_path, "rb")
    #             dic_to_merge = pickle.load(pickle_file)
    #             pickle_file.close()
    #             for key in dic_to_merge:
    #                 if key in merged_data_dic:
    #                     merged_data_dic[key][0].update(dic_to_merge[key][0])
    #                     merged_data_dic[key][1] += dic_to_merge[key][1]
    #
    #                 else:
    #                     merged_data_dic[key] = dic_to_merge[key]
    #         else:
    #             self.update_dic(merged_data_dic, file_name_to_be_merged)
    #             return merged_data_dic
    #         os.remove(file_path)
    #     self.update_dic(merged_data_dic, file_name_to_be_merged)
    #     return merged_data_dic
    #
    # def update_dic(self, merged_data_dic, file_name_to_be_merged):
    #     """
    #     This function merge two dictionaries
    #     :param merged_data_dic: dictionary to merge to
    #     :param file_name_to_be_merged: letter we merge
    #     :return:
    #     """
    #     if bool(self.capital_letters_dic[file_name_to_be_merged]):
    #         for item in self.capital_letters_dic[file_name_to_be_merged]:
    #             if item in merged_data_dic:
    #                 merged_data_dic[item][0].update(self.capital_letters_dic[file_name_to_be_merged][item][0])
    #                 merged_data_dic[item][1] += self.capital_letters_dic[file_name_to_be_merged][item][1]
    #
    #             else:
    #                 merged_data_dic[item] = self.capital_letters_dic[file_name_to_be_merged][item]

    def parse_url(self, url_term):
        """
        :param url_term: string,an url term
        :return : list of url tokens
        """
        if url_term is None:
            return ""
        tokens_list = []
        url_parsed = urlparse(url_term)  # parse url
        # tokens_list.append(url_parsed.scheme)  # append internet protocol
        address = url_parsed.netloc
        if 'www' in address:
            # tokens_list.append('www')
            address = re.sub('www.', '', address)

        tokens_list.append(address)  # delete from url internet protocol and address
        rest_of_url = url_term.replace(url_parsed.scheme + '://' + url_parsed.netloc, '')
        url_to_kens = rest_of_url.split('/')
        for item in url_to_kens:
            splitted_items = split_string(item, '[`!~&*=?|;#%.+_,(){}]')  # split each item by these delimiters
            if len(splitted_items) == 0:
                continue
            for splitedItem in splitted_items:
                if splitedItem == '':
                    continue
                tokens_list.append(splitedItem)
        return tokens_list
