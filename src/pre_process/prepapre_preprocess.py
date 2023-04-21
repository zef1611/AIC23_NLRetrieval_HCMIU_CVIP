# created by Nguyen Qui Vinh Quang at 2022-03-10 16:28.
#
# @Property of CVIPLab-2012
# @Contest: AIC2022
# Jisoo so cute

import json
from SRL import SRL
import spacy
import os
from tqdm import tqdm
import ROOT_PATH

from text_extraction.SubjectExtraction import SubjectExtraction
from text_extraction.VehicleAroundExtraction import VehicleAroundExtraction
from text_extraction.DirectionExtraction import DirectionExtraction
from pre_config import get_default_config
import argparse

root = os.path.join(ROOT_PATH.root, "codebook")


class PreparePreprocess(object):
    def __init__(self, cfg, srl=None, spacy_nlp=None):
        self.cfg = cfg
        if srl == None:
            # print(os.getcwd())
            # exit(0)
            self.srl = SRL(
                "src/pre_process/weight/structured-prediction-srl-bert.2020.12.15.tar.gz"
            )
        else:
            self.srl = srl

        if spacy_nlp == None:
            self.nlp = spacy.load("en_core_web_sm")
        else:
            self.nlp = spacy_nlp

        self.subjectExtraction = SubjectExtraction(cfg, self.srl, self.nlp)
        self.vehicleArround = VehicleAroundExtraction(cfg, self.subjectExtraction, self.nlp)
        self.directionExtraction = DirectionExtraction(cfg, self.nlp)

        return

    def prepareTrain(self, year=2022):
        print("Starting at train file")
        f = open(self.cfg.DATA.TRAIN_FILES_2022)
        f = json.load(f)
        sentences = set()
        for k in f:
            for s in f[k]["nl"]:
                sentences.add(s)
            for s in f[k]["nl_other_views"]:
                sentences.add(s)
        subject = {}
        direction = {}
        arround = {}
        for (_, s) in tqdm(enumerate(sentences)):
            subject[s] = self.subjectExtraction.extractSubject(s)["subject"]
            _arround = self.vehicleArround.process(s)
            if _arround:
                arround[s] = _arround
            direction[s] = self.directionExtraction.findDirections(s)

        subject_filename = "train-subjects-{}.json".format(str(year))
        arround_filename = "train-around-extraction-{}.json".format(str(year))
        direction_filename = "train-directions-{}.json".format(str(year))

        save_arr = [
            [subject_filename, subject],
            [arround_filename, arround],
            [direction_filename, direction],
        ]

        for (save_file, _dict) in save_arr:
            save_file = os.path.join(ROOT_PATH.root, "json_folders", save_file)
            with open(save_file, "w") as outfile:
                json.dump(_dict, outfile, indent=2)

        return

    def prepareTest(self, year=2022):
        print("Starting at test file")
        f = open(self.cfg.DATA.TEST_FILES_2022)
        f = json.load(f)
        sentences = set()
        for k in f:
            nl_array = f[k]["nl"]
            for v in f[k]["nl_other_views"]:
                nl_array.append(v)
            for s in nl_array:
                sentences.add(s)
        subject = {}
        direction = {}
        arround = {}
        for (_, s) in tqdm(enumerate(sentences)):
            subject[s] = self.subjectExtraction.extractSubject(s)["subject"]
            if s == "A black pickup is behind another black pickup.":
                print("aa")
            _arround = self.vehicleArround.process(s)
            if _arround:
                arround[s] = _arround
            direction[s] = self.directionExtraction.findDirections(s)

        subject_filename = "test-subjects-{}.json".format(str(year))
        arround_filename = "test-around-extraction-{}.json".format(str(year))
        direction_filename = "test-directions-{}.json".format(str(year))

        save_arr = [
            [subject_filename, subject],
            [arround_filename, arround],
            [direction_filename, direction],
        ]

        for (save_file, _dict) in save_arr:
            save_file = os.path.join(ROOT_PATH.root, "json_folders", save_file)
            with open(save_file, "w") as outfile:
                json.dump(_dict, outfile, indent=2)

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AICT5 prepare-preprocess")
    
    parser.add_argument(
        "--list_type",
        type=str,
        nargs="+",
        help="Valid arguments: train_2022, test_2022",
    )

    args = parser.parse_args()
    print(args.list_type)
    cfg = get_default_config()

    prepare = PreparePreprocess(cfg)
    for _data_type in args.list_type:
        print("Process data:", _data_type)
        if _data_type == "train_2022":
            prepare.prepareTrain(year=2022)
        elif _data_type == "test_2022":
            prepare.prepareTest(year=2022)
        else:
            print("Does not support")