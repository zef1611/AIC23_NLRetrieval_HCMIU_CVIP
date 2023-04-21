from TurnDetector import *
from StopDetector import *
import json

full_uuid = json.load(open("./data/json/original_data/test-tracks.json"))
cur_result = json.load(open("./data/json/recognition/test-tracks-direction.json"))

stop_detector = StopDetector()

def run_direction(cur_result):
    for query in full_uuid:
        turn_state = get_directions(full_uuid[query]["boxes"])
        if query not in cur_result or cur_result[query]["id"] != 0:
            cur_result[query] = dict()
            cur_result[query]["id"] = [turn_state]
        else:
            cur_result[query]["id"] = [cur_result[query]["id"]]
            if turn_state != 0:
                cur_result[query]["id"].append(turn_state)
        if stop_detector.process(full_uuid[query]["boxes"]):
            cur_result[query]["id"].append(3)
    json.dump(
        cur_result,
        open("./data/json/recognition/test-tracks-direction-refinement.json", "w"),
        indent=4,
    )

run_direction(cur_result)