import json

directions_text_dict = json.load(open("./data/json/recognition/test-queries-direction.json"))
colors_text_dict = json.load(open("./data/json/recognition/test-queries-color.json"))
type_text_dict = json.load(open("./data/json/recognition/test-queries-type.json"))

attr_text = {
    "directions": directions_text_dict,
    "colors": colors_text_dict,
    "type": type_text_dict,
}

def is_same(attr_text, attr_track):
    for attr in attr_track:
        if attr_track[attr] != attr_text[attr]:
            return False
    return True

def get_attr(u, colors_dict, type_dict, direction_dict):

    color = colors_dict[u]["id"]
    type = type_dict[u]["id"]
    direction = direction_dict[u]["id"]
    attr_track = {"colors": color, "type": type}

    return attr_track, direction


def Rerank(result_data, type_dict, colors_dict, directions_dict):
    final_rank = {}

    for query in result_data:
        potential_cand = []
        potential2_cand = []
        potential3_cand = []
        unpotential_cand = []

        for track in result_data[query]:
            track_attr, track_direction = get_attr(
                track, colors_dict, type_dict, directions_dict
            )
            text_attr, text_direction = get_attr(
                query, colors_text_dict, type_text_dict, directions_text_dict
            )
            if not is_same(text_attr, track_attr):
                unpotential_cand.append(track)
            else:
                if track_direction == text_direction:
                    potential_cand.append(track)
                else:
                    inside = False
                    for direction in track_direction:
                        if direction in text_direction:
                            potential2_cand.append(track)
                            inside = True
                            break
                    if not inside:
                        potential3_cand.append(track)
        potential_cand = potential_cand + potential2_cand + potential3_cand
        final_rank[query] = potential_cand + unpotential_cand

    return final_rank