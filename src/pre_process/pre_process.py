# created by Nguyen Qui Vinh Quang at 2022-02-10 11:59.
#
# @Property of CVIPLab-2012
# @Contest: AIC2022
# Jisoo so cute

import re
import enchant
from matplotlib.pyplot import subplots_adjust
from nltk.stem.wordnet import WordNetLemmatizer
import json
import os
from torch import save, true_divide
from tqdm import tqdm
import ROOT_PATH

root = ROOT_PATH.root

ROMOVE_WORLD = [
    "chevy",
]


class PreProcess(object):
    def __init__(
        self,
        color_filename="valid-color_v2.txt",
        vehicle_filename="valid-vehicle_v2.txt",
        replace_filename="replace_words.txt",
        train=True,
        year=2021,
    ):
        # filenames
        self.color_filename = os.path.join(root, "codebook", color_filename)
        self.vehicle_filename = os.path.join(root, "codebook", vehicle_filename)
        self.replace_filename = os.path.join(root, "codebook", replace_filename)

        # word define
        self.colors = set()
        self.vehicles = set()
        self.english_dictionary = enchant.Dict("en_US")
        self.hash_color = self.read_file(self.color_filename)
        for color in self.hash_color:
            self.colors.add(color)
        self.words = self.read_file(self.replace_filename)
        self.hash_vehicle = self.read_file(self.vehicle_filename)
        for vehicle in self.hash_vehicle:
            self.vehicles.add(vehicle)

        self.arg = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"]

        self.train = train  # train or test
        self.__preprocess(year)

        return

    def read_file(self, filename):
        f = open(filename)
        f = f.readlines()
        _hash = {}

        for line in f:
            line = line.replace("\n", "")
            key, value = line.split(": ")
            key = " " + key + " "
            value = " " + value + " "
            _hash[key] = value

        _hash = dict(sorted(_hash.items(), key=lambda item: -len(item[0])))
        return _hash

    def __preprocess(self, year):
        if self.train == True:
            f = open(
                os.path.join(
                    root, "json_folders/train-subjects-{}.json".format(str(year))
                )
            )
            self.subjects = json.load(f)

            f = open(
                os.path.join(
                    root, "json_folders/train-directions-{}.json".format(str(year))
                )
            )
            self.directions = json.load(f)

            f = open(
                os.path.join(
                    root,
                    "json_folders/train-around-extraction-{}.json".format(str(year)),
                )
            )
            self.vehicleAround = json.load(f)
        else:
            f = open(
                os.path.join(
                    root, "json_folders/test-subjects-{}.json".format(str(year))
                )
            )
            self.subjects = json.load(f)

            f = open(
                os.path.join(
                    root, "json_folders/test-directions-{}.json".format(str(year))
                )
            )
            self.directions = json.load(f)

            f = open(
                os.path.join(
                    root,
                    "json_folders/test-around-extraction-{}.json".format(str(year)),
                )
            )
            self.vehicleAround = json.load(f)
        return

    def convert_sentence_to_word_list(self, lst):
        # Removing punctuations in string
        # Using regex
        lst = re.sub(r"[^\w\s]", "", lst)
        return lst.split(" ")

    def replace_word(self, query, _dict):
        for w in _dict:
            query = query.replace(w, _dict[w])

        return query

    def word_distance(self, str1, str2):
        return enchant.utils.levenshtein(str1, str2)

    def isError(self, word):
        """return True if word is not in Dictionary."""
        if word in self.hash_color or len(word) == 0 or word in self.hash_vehicle:
            """
            If the word is the vehicle name,
            it not consider as the spelling mistake

            """
            return False
        return not self.english_dictionary.check(word)

    def convert_words_To_Sentence(self, words):
        str = ""
        for w in words:
            str = str + " " + w

        return str[1:]

    def __removeTheLastSpace(self, sentence):
        """Remove all spaces at the end of sentence"""
        while len(sentence) > 1 and sentence[-1] == " ":
            sentence = sentence[:-1]
        return sentence

    def __removeTheFirstSpace(self, sentence):
        # reverse the string and remove the last space
        sentence = self.__removeTheLastSpace(sentence[::-1])
        # reverse to the original string
        sentence = sentence[::-1]
        return sentence

    def cleanSentence(self, sentence):
        sentence = sentence.replace("A ", "")
        sentence = sentence.replace("An ", "")
        sentence = sentence.replace("The ", "")

        sentence = self.__removeTheFirstSpace(sentence)
        sentence = self.__removeTheLastSpace(sentence)
        sentence = sentence.replace("  ", " ")
        # remove the dot
        if sentence[-1] == ".":
            sentence = sentence[:-1]
        return sentence.lower()

    def fixError_word(self, word):
        """If the word has a spelling error, this function compares
        the word with my custom dictionary (colour, vehicle type) and
        find the word with the smallest levenshtein distance. If not, return itself

        Args:
            word ([str]): a single word
        Returns:
            [str]: the word with the smallest levenshtein or itself
        """
        distance = {}

        for val in self.colors:
            distance[val] = self.word_distance(word, val)
        for val in self.vehicles:
            distance[val] = self.word_distance(word, val)

        min_val = 1e9
        key = None
        for _key in distance:
            if distance[_key] < min_val:
                min_val = distance[_key]
                key = _key
        if min_val > len(word):
            key = word
        return key

    def fixError_sentence(self, sentence):
        """Correct spelling mistakes in sentences.
        Args:
            sentence ([str]): One sentece.

        Returns:
            [str]: sentence
        """
        sentence = self.convert_sentence_to_word_list(sentence)
        for idx in range(len(sentence)):
            if self.isError(sentence[idx]):
                sentence[idx] = self.fixError_word(sentence[idx])
            else:
                if sentence[idx] == "left":
                    continue
                sentence[idx] = WordNetLemmatizer().lemmatize(sentence[idx], "v")
        return self.convert_words_To_Sentence(sentence)

    def sentenceSubjectDirection(self, sentence, subject=None):
        if subject == None:
            subject = self.subjects[sentence]
            # subject = ""
        # if subject == None:
        #     return '_.'
        direction = self.directions[sentence]
        d = {}
        for val in direction:
            if direction[val] == -1:
                continue
            d[val] = direction[val]
        if d == {}:
            return subject + ".", False
        d = dict(sorted(d.items(), key=lambda item: len(item[0])))
        for val in d:
            subject = subject + " " + val
        # subject = subject
        return subject, True

    def sentenceVehicleAround(self, sentence):
        if sentence not in self.vehicleAround:
            return "_."
        res = ""
        for v in self.vehicleAround[sentence]:
            res = res + self.vehicleAround[sentence][v] + " "
        res = res[:-1] + "."
        return res

    def process(self, _query, format="standardized"):
        # print(_query)
        """With each string input, I will change some words to my cumtome dictionary. It also
            corrects spelling mistake.
            Example:
                Quyên was the love of  my life -> Quyên is the love of my life.
                1 car -> one car.
                A silver SUV follows a black SUV down the street -> A grey SUV follows a black SUV down the street

        Args:
            query ([type]): input sentence
            format([str]): + normalize
                           + clean

        Returns:
            [str]: the sentence after process
        """
        # query = self.replace_word(query, self.hash_color)
        query = self.replace_word(_query, self.words)
        query = self.fixError_sentence(query)
        query = self.cleanSentence(query)
        if format == "clean":
            return query, True
        # return query +' ', True
        # query = query + '. ' + \
        #     self.sentenceSubjectDirection(
        #         _query) + ' ' + self.sentenceVehicleAround(_query)
        query, hasDirection = self.sentenceSubjectDirection(_query)
        if query == "_.":
            query = ""
        # if 'intersection' in _query or 'intersection.' in _query:
        #     query = query.replace('.',' ')
        #     query = query + 'intersection'
        return query, hasDirection


def process(filepath, year="2022", save_folder=None, format_type="clean", isTrain=True):
    """prepapre train

    Args:
        year (str, optional):
            + Year of data use to preprocess.
            + 2022
        format_type (str, optional):
            + clean
            + standardized
        isTrain:
            + True: text is used for train
            + False: text is used for test/val
        save_folder(str): The path of folder you want to save files
    """
    pre = PreProcess(train=isTrain, year=year)

    f = json.load(open(filepath))
    _temp = {}
    for (_, key) in tqdm(enumerate(f)):
        strs = f[key]["nl"]
        ans = []
        if key == '890d859a-3de8-43f5-9c43-29f8ffcbc55d':
            print('ok')
        for s in strs:
            _s, hasDirection = pre.process(s, format=format_type)
            _temp[s] = _s
            s = _s
            if s == "" or hasDirection == False:
                continue
            # if 'go' not in s and 'turn' not in s: continue
            s = s.replace("  ", " ")
            ans.append(s)

        if ans == []:
            ans = ["vehicle go straight"]
        f[key]["nl"] = ans
    save_file = "pre_process/data/tune-convert-train-2021-aug.json"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if isTrain:
        filename = 'train_{}_{}.json'.format(str(year), format_type)
    else:
        filename = 'eval_{}_{}.json'.format(str(year), format_type)
    save_file = os.path.join(save_folder,filename)
    
    with open(save_file, "w") as outfile:
        json.dump(f, outfile, indent=4)
    print(save_file)

    return


def train(year="2022", save_folder=None, format_type="clean"):
    filepath = "data/json/original_data/train-tracks.json"
    process(filepath=filepath, year=year, save_folder=save_folder,format_type=format_type, isTrain=True)
    return


def test(year="2022", save_folder=None, format_type="clean"):
    filepath = "data/json/original_data/test-queries.json"
    process(filepath=filepath, year=year,save_folder=save_folder, format_type=format_type, isTrain=False)
    return


if __name__ == "__main__":
    save_folder = 'data/json/dataclean_v1/'

    train(save_folder=save_folder,format_type="clean")
    train(save_folder=save_folder,format_type="standardized")
    # test(save_folder=save_folder,format_type="clean")
    # test(save_folder=save_folder,format_type="standardized")

    print("finish preprocess")
