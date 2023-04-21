# created by Nguyen Qui Vinh Quang at 2022-08-24 09:09.
#
# @Property of CVIPLab-2012
# Jisoo so cute
import os
import json
import string
import nltk
import re
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from multiprocessing import Pool
from pre_config import get_default_config
stopwords = nltk.corpus.stopwords.words("english")


class Clean_text(object):
    def __init__(self, cfg, json_clean_path, threads=2):

        _path = os.path.join(cfg.DATA.ROOT, "codebook")
        self.ignore_words = set()
        self.convert_pharse = dict()
        self.threads = threads
        self.read_file_convert(os.path.join(_path, cfg.DATA.COLOR_FILENAME))
        self.read_file_convert(os.path.join(_path, cfg.DATA.VEHICLE_FILENAME))
        self.read_file_convert(os.path.join(_path, cfg.DATA.DIRECTION_FILENAME))

        self.read_ignore_words(os.path.join(_path, cfg.DATA.VEHICLE_FILENAME))
        self.read_ignore_words(os.path.join(_path, cfg.DATA.DIRECTION_FILENAME))

        self.json_files = json.load(open(json_clean_path))
        
        raw_sentences = set()
        for uuid in self.json_files:
            for s in self.json_files[uuid]['nl']:
                raw_sentences.add(s)

        self.raw_sentences = list(raw_sentences)

        if not os.path.exists(cfg.DATA.OUTPUT):
            os.makedirs(cfg.DATA.OUTPUT)
        return

    def func(self, s):
        _clean = self.preprocess(s)
        return (s, _clean)

    def process(self):
        with Pool(self.threads) as p:
            temp = p.map(self.func, self.raw_sentences)
        print(len(temp))
        sentences = {}
        for (ori, convert) in temp:
            convert = convert.replace('intervention', 'intersection')
            sentences[ori] = [convert]
        with open(os.path.join(cfg.DATA.OUTPUT, "sentence-clean.json"), 'w') as outfile:
            json.dump(sentences, outfile, indent=4)
        return


    def read_file_convert(self, filename: str):
        f = open(filename, "r")
        f = f.readlines()
        for line in f:
            ori, convert = line.split(":")
            convert = convert.replace("\n", "")
            self.convert_pharse[ori] = convert

        return

    def read_ignore_words(self, filename):
        f = open(filename, "r")
        f = f.readlines()
        for line in f:
            ori, convert = line.split(":")
            convert = convert.replace("\n", "")
            for w in re.split(" +", convert):
                self.ignore_words.add(w)

        return

    def replace_words(self, text):
        for phrase in self.convert_pharse:
            if phrase in text:
                text = text.replace(phrase, self.convert_pharse[phrase])
        return text

    def remove_punctuation(self, text):
        punctuationfree = "".join([i for i in text if i not in string.punctuation])
        return punctuationfree

    def lower_text(self, text: str):
        text = text.lower()
        return text

    def tokenization(self, text):

        tokens = re.split(" +", text)
        return tokens

    def remove_stopwords(self, text):
        output = [i for i in text if (i not in stopwords or i == 'up' or i == 'down')]
        return output

    def spelling_correction(self, tokens):
        for (idx, w) in enumerate(tokens):
            if w in self.ignore_words:
                continue
            if "left" in w:
                print(w)
            tokens[idx] = str(TextBlob(w).correct())

        return tokens

    def Lemmatization(self, token_text):
        wordNetLemmatizer = WordNetLemmatizer()

        output = [
            wordNetLemmatizer.lemmatize(word, pos="v")
            if word not in self.ignore_words
            else word
            for word in token_text
        ]
        return output

    def preprocess(self, text):
        # text = self.replace_words(text)
        text = self.remove_punctuation(text)
        text = self.lower_text(text)
        tokens = self.tokenization(text)
        tokens = self.remove_stopwords(tokens)
        # tokens = self.spelling_correction(tokens)
        # tokens = self.Lemmatization(tokens)

        output = "".join(token + " " for token in tokens)
        return output[:-1]

if __name__ == '__main__':
    """
        Test function
    """
    cfg = get_default_config()
    
    clean = Clean_text(cfg,'pre_process/json_folders/train_tracks.json',threads=6 )
    clean.process()