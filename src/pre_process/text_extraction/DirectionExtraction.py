# created by Nguyen Qui Vinh Quang at 2022-02-27 22:41.
#
# @Property of CVIPLab-2012
# @Contest: AIC2022
# Jisoo so cute
import json
import spacy
from nltk.stem.wordnet import WordNetLemmatizer
import os
import ROOT_PATH as ROOT_PATH
from tqdm import tqdm
from pre_config import get_default_config

# root = 'pre_process/codebook'
cfg = get_default_config()
root = os.path.join(str(ROOT_PATH.root), "codebook")


class DirectionExtraction(object):
    def __init__(self, args,  nlpSpacy=None):
        self.valid_directions = {"left", "right", "stop", "wait"}

        if nlpSpacy == None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = nlpSpacy
        # root = "codebook"
        path = os.path.join(root, args.DATA.DIRECTION_FILENAME)
        self.directions = self.readValidWords(path)
        return

    def readValidWords(self, filename):
        file = open(filename, "r")
        f = file.readlines()
        _dict = {}
        for line in f:
            old_word, new_word = line.split(": ")
            new_word = new_word.replace("\n", "")
            old_word, new_word = old_word.lower(), new_word.lower()
            _dict[old_word] = new_word

        _dict = dict(sorted(_dict.items(), key=lambda item: -len(item[0])))

        return _dict

    def specialCase(sentence, occur):
        """this function focus to handle the case: do sth A after do sth B
        Ex:
            The vechile stops at the intersection **after** turn left
            -> return: turn left: 1, stop: 2
        Args:
            sentence (string): sentence
            occur (dict): the occurence postion, input must the sorted by value
        """
        if "after " not in sentence:
            return occur
        pos = sentence.find("after ")
        forward = []
        backward = []
        # convert dict to list

        for val in occur:
            if occur[val] < pos:
                forward.append((val, occur[val]))
            else:
                backward.append((val, occur[val]))
        _res = {}
        total = 1
        for v in backward:
            if v[1] == -1:
                _res[v[0]] = v[1]
            else:
                _res[v[0]] = total
                total += 1
        for v in forward:
            if v[1] == -1:
                _res[v[0]] = v[1]
            else:
                _res[v[0]] = total
                total += 1

        return _res

    def findDirections(self, sentence):
        """List the order of movement direction of the vehicle
        Args:
            sentence (str): sentence
        Returns:
            dict: {direction, order}
            ex: _direction = {'turn left': -1, 'turn right': -1,
                                'go straight': -1, 'stop': -1, 'special':-1 }
        """
        _direction = {
            "turn left": -1,
            "turn right": -1,
            "go straight": -1,
            "stop": -1,
            "special": -1,
        }
        text = self.nlp(sentence)
        sentence = ""
        for token in text:
            w = WordNetLemmatizer().lemmatize(token.text, "v")
            if token.text == "left":
                w = "left"
            sentence = sentence + w + " "
        sentence = sentence[:-1]
        sentence = sentence.replace(" - ", "-")

        occurList = {}
        for direction in self.directions:
            pos = sentence.find(direction)
            if pos == -1:
                continue
            occurList[self.directions[direction]] = pos

        occurList = dict(sorted(occurList.items(), key=lambda item: item[1]))

        for (idx, direc) in enumerate(occurList):
            _direction[direc] = idx + 1

        # Find occur exist any direction in sentence
        # If not -> Vehicle go straight
        _check = False
        for _direc in _direction:
            if _direction[_direc] != -1:
                _check = True
                break
        if _check == False:
            _direction["go straight"] = 1
        return _direction


if __name__ == "__main__":
    # prepareTrain()
    # prepareTest()
    # print(direction.findDirections('A white sedan follows a black pick up truck.'))
    directionExtraction = DirectionExtraction(cfg)
    # print(directionExtraction.findDirections("A white sedan is behind a pickup truck."))
    filename = 'data/json/uuid/tune.json'
    f = json.load(open(filename))
    sentences = set()
    for k in f:
        for s in f[k]['nl']:
            sentences.add(s)
    direction = {}
    for (_, s) in tqdm(enumerate(sentences)):
        direction[s] = directionExtraction.findDirections(s)

    with open('data/json/uuid/directions.json', 'w') as outfile:
        json.dump(direction, outfile, indent=2)
