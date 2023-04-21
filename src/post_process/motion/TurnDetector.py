import json
import numpy as np

direction_id = json.load(open('./data/json/recognition/direction_id.json'))

def get_directions(list_boxes):

    list_points = []

    for box in list_boxes:
        list_points.append([box[0] + box[2] / 2, box[1] + box[3] / 2])

    list_points = np.array(list_points)

    v_aa1 = list_points[1]  - list_points[0]
    v_aan = list_points[-1] - list_points[0]
    d = np.cross(v_aa1, v_aan)

    if d < 0:
        return direction_id['turn left']
    return direction_id['turn right']