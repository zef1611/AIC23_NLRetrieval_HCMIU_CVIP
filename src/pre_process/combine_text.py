"""
    Combine full text and standardize test
"""
import json
import os

full_text = json.load(open('pre_process/data/dataclean/eval-clean.json'))
standard_text = json.load(open('pre_process/data/dataclean/eval-standard.json'))


for uuid in full_text:
    for sen in standard_text[uuid]['nl']:
        full_text[uuid]['nl'].append(sen)

with open("pre_process/data/dataclean/eval-combine.json", "w") as outfile:
    json.dump(full_text, outfile, indent=2)
