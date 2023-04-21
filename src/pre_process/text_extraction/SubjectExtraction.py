# created by Nguyen Qui Vinh Quang at 2022-02-25 00:08.
#
# @Property of CVIPLab-2012
# @Contest: AIC2022
# Jisoo so cute
import sys
# sys.path.append('../')
import json
from multiprocessing.dummy import Pool
import re
from turtle import color
from  SRL import SRL
import spacy
import os
from tqdm import tqdm
import ROOT_PATH as ROOT_PATH
from pre_config import get_default_config
root = os.path.join(ROOT_PATH.root, "codebook")

print('ok')
class SubjectExtraction(object):
    def __init__(self,args,srl=None, spacy_nlp=None):

        # define file path
        self.root = root
        self.args = args
        self.color_filename = os.path.join(root, args.DATA.COLOR_FILENAME)
        self.vehicle_filname = os.path.join(root, args.DATA.VEHICLE_FILENAME)
        self.size_filename = os.path.join(root, args.DATA.SIZE_FILENAME)

        # define some specific variable
        self.arg = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"]
        self.article = {"a", "an", "the", "A", "An", "The", "Aa"}
        self.replace_word = {
            "cross-over": "crossover",
            "cross over": "crossover",
            "pickup truck": "pickup-truck",
            ".": "",
        }
        self.replace_character = {
            "\n": "",
            "  ": " ",
            " - ": "-",
        }
        # load srl:
        if srl == None:
            self.srl = SRL(
                "src/pre_process/weight/structured-prediction-srl-bert.2020.12.15.tar.gz"
            )
        else:
            self.srl = srl

        # load spacy
        if spacy_nlp == None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = spacy_nlp
        # preprocess vehicle
        self.replace_vehicle = self.readValidWords(self.vehicle_filname)
        valid_vehicle = []
        for c in self.replace_vehicle:
            valid_vehicle.append(self.replaceLastSpace(self.replace_vehicle[c]))
            self.replace_word[c] = self.replace_vehicle[c]
        self.valid_vehicle = valid_vehicle

        # preprocess color
        self.replace_color = self.readValidWords(self.color_filename)
        valid_color = []
        for c in self.replace_color:
            valid_color.append(self.replace_color[c])
            self.replace_word[c] = self.replace_color[c]
        self.valid_color = valid_color

        # preprocess size
        self.replace_size = self.readValidWords(self.size_filename)
        valid_size = []
        for c in self.replace_size:
            valid_size.append(self.replace_size[c])
            self.replace_word[c] = self.replace_size[c]
        self.valid_size = valid_size

        return

    def replaceLastSpace(self, sen):
        while len(sen) > 0 and sen[-1] == " ":
            sen = sen[:-1]
        return sen

    def convert_sentence_to_word_list(self, lst):
        # Removing punctuations in string
        # Using regex
        lst = lst.replace(" - ", "-")
        lst = lst.replace("  ", " ")
        lst = lst.split(" ")
        lst.remove("")
        return lst

    def readValidWords(self, filename):
        file = open(filename, "r")
        f = file.readlines()
        _dict = {}
        for line in f:
            old_word, new_word = line.split(": ")
            new_word = new_word.replace("\n", "")

            new_word = self.replaceLastSpace(new_word)
            old_word = self.replaceLastSpace(old_word)
            old_word = old_word + " "
            new_word = new_word + " "
            old_word, new_word = old_word.lower(), new_word.lower()
            _dict[old_word] = new_word
        _dict = dict(sorted(_dict.items(), key=lambda item: -len(item[0])))
        return _dict

    def findValInSubject(self, subject, arr):
        """
        find some value in arr exist in subject
        """
        for val in arr:
            if val in subject:
                return val

        return None

    def findColor(self, subject):
        """
        find color occur in main subject
        """
        return self.findValInSubject(subject, self.valid_color)

    def findVehicle(self, subject):
        """
        find vechile occur in main subject
        """
        return self.findValInSubject(subject, self.valid_vehicle)

    def findSize(self, subject):
        """find size of vehicle"""
        return self.findValInSubject(subject, self.valid_size)

    def cleanMainSubject(self, subject):
        firstWord = ""
        result = ""
        ok = False
        subject = subject + " "
        # Remove article
        for w in subject:
            if w == " " and ok == False:
                ok = True
                firstWord = firstWord.lower()
                if firstWord in self.article:
                    continue
                result = firstWord + " "
            if ok == False:
                firstWord = firstWord + w
            else:
                result = result + w.lower()

        # clean some character
        for character in self.replace_character:
            result = result.replace(character, self.replace_character[character])

        # remove the last space in subject
        if len(result) > 1 and result[-1] == " ":
            result = result[:-1]
        # remove the first space in subject
        while len(result) > 1 and result[0] == " ":
            result = result[1:]
        return result

    def findMainSubjectSRL(self, sentence):
        """Find the main subject using SRL. The main subject is
           the first ARG.
        Args:
            sentence (string): the input sentence

        Returns:
            + string: the main subject
            + None: if cannot find main subject
        """
        query = self.srl.extract(sentence)
        
        for verb in query["verbs"]:
            out = re.findall(r"\[(.*?)\]", verb["description"])
            for ans in out:
                argument, content = ans.split(": ")
                if argument in self.arg and self.findVehicle(argument) != None:
                    return content
        return None

    def findMainSubjectHeuristic(self, sentence):
        """Find main subject. I assume that the mainsubject is start from the
        first word to the word which has the vechile meaning.

        Args:
            sentence (string): the input sentence

        Returns:
            + string: the main subject
            + None: if cannot find main subject
        """
        out = self.nlp(sentence)
        mainSub = ""
        arr = []
        for (idx, token) in enumerate(out):
            arr.append(token.text)
            if token.text.lower() in self.valid_vehicle:
                break
    

        for (idx, s) in enumerate(arr):
            if idx == 0 and s in self.article:
                continue
            mainSub = mainSub + s.lower() + " "

        return mainSub

    def findSubject(self, sentence):
        if sentence == 'A brown SUV runs down the street followed by another gray sedan.':
            print("here")
        # Clean subject
        while sentence[:-1] == ".":
            sentence = sentence[:-1]
        sentence = sentence + " "
        sentence = sentence.lower()
        for w in self.replace_word:
            sentence = sentence.replace(w, self.replace_word[w])

        subject = self.findMainSubjectSRL(sentence)
        if subject != None:
            return self.cleanMainSubject(subject)

        subject = self.findMainSubjectHeuristic(sentence)

        if subject != None:
            return self.cleanMainSubject(subject)
        return None

    def extractSubject(self, sentence):
        _dict = {"subject": None, "color": None, "vehicle": None, "size": None}
        sub = self.findSubject(sentence)
        if sub == None:
            return _dict
        _dict["subject"] = sub
        _dict["color"] = self.findColor(sub)
        _dict["vehicle"] = self.findVehicle(sub)
        _dict["size"] = self.findSize(sub)
        return _dict


if __name__ == "__main__":
    cfg = get_default_config()
    subE = SubjectExtraction(cfg)
    f = open("data/json/uuid/tune.json")
    f = json.load(f)

    sentences = set()
    for key in  f:
        for s in f[key]['nl']:
            sentences.add(s)
    # del f
    subject = {}
    def f(s):
        subject[s] = subE.extractSubject(s)
    with Pool(3) as p:
        p.map(f, list(sentences))
    print(len(subject)) 
    with open('data/json/uuid/type_color_direction_train.json', 'w') as outfile:
        json.dump(subject, outfile, indent=2)
    
