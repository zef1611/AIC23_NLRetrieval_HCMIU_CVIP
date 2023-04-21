# created by Nguyen Qui Vinh Quang at 2022-02-28 01:39.
#
# @Property of CVIPLab-2012
# @Contest: AIC2022
# Jisoo so cute

import spacy
from text_extraction.SubjectExtraction import SubjectExtraction
import os
import json
import ROOT_PATH as ROOT_PATH
from pre_config import get_default_config
from multiprocessing.dummy import Pool

# root = 'pre_process/codebook/'
# root = 'codebook/'
root = os.path.join(ROOT_PATH.root, "codebook")


class VehicleAroundExtraction(object):
    def __init__(self, cfg, subE=None, nlp=None, followFilename="valid-follows.txt"):
        if subE == None:
            self.getSubject = SubjectExtraction(cfg)
        else:
            self.getSubject = subE
        if nlp == None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlp

        self.follow = self.readValidWords(os.path.join(root, followFilename))
        self.deleteWords = {}
        return

    def readValidWords(self, filename):
        file = open(filename, "r")
        f = file.readlines()
        _dict = {}
        for line in f:
            old_word, new_word = line.split(": ")
            new_word = new_word.replace("\n", "")
            old_word = " " + old_word + " "
            old_word, new_word = old_word.lower(), new_word.lower()
            _dict[old_word] = new_word
        _dict = dict(sorted(_dict.items(), key=lambda item: -len(item[0])))

        return _dict

    def readDelWord(self, filename="delete_words-directions.txt"):
        filename = root + filename
        file = open(filename, "r")
        f = file.readlines()
        _delW = set()
        for line in f:
            line = line.replace("\n", "")
            _delW.add(line)
        return _delW

    def remove_redundant_words(self, subject: str):
        if subject == None:
            return subject
        subject = " " + subject + " "
        for w in self.deleteWords:
            subject = subject.replace(w, "")
        # clean words
        while len(subject) > 0 and subject[0] == " ":
            subject = subject[1:]
        while len(subject) > 0 and subject[-1] == " ":
            subject = subject[:-1]

        return subject

    def getVehicle(self, sentence, kw):
        try:
            fiSen, seSen = sentence.split(kw)
        except:
            print(sentence)
            return None, None
        firtSub = self.getSubject.extractSubject(fiSen)["subject"]
        seSub = self.getSubject.extractSubject(seSen)["subject"]

        if firtSub == None or seSub == None:
            return None, None

        return self.remove_redundant_words(firtSub), self.remove_redundant_words(seSub)

    def process(self, sentence):
        ans = None
        for f in self.follow:
            if f in sentence:
                firstV, secondV = self.getVehicle(sentence, f)
                if firstV == None:
                    continue
                ans = {}
                ans["A"] = firstV
                ans["follow"] = self.follow[f]
                ans["B"] = secondV
                return ans
        return ans


if __name__ == "__main__":
    cfg = get_default_config()
    testcase = VehicleAroundExtraction(cfg)
    ex = testcase.process("A black pickup is behind another black pickup.")
    print(ex)
    f = open("eval_clean.json")
    f = json.load(f)
    # print("Vehicle arround extraction finish")
    # prepareTrain()
    sentences = set()
    for key in  f:
        for s in f[key]['nl']:
            sentences.add(s)
    # del f
    subject = {}
    def f(s):
        subject[s] = testcase.process(s)
    with Pool(3) as p:
        p.map(f, list(sentences))
    print(len(subject)) 
    with open('relation.json', 'w') as outfile:
        json.dump(subject, outfile, indent=2)