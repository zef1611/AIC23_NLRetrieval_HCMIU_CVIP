import os
import json
TEXT_FILE = 'data/json/uuid/tune.json'

DIRECTION_PATH = 'data/json/uuid/directions.json'
SUBJECT_PATH = 'data/json/uuid/type_color_direction_train.json'

direction = json.load(open(DIRECTION_PATH))
subject = json.load(open(SUBJECT_PATH))


_data = json.load(open(TEXT_FILE))
_output = {}
def direction_process(direc):
    lst = []
    for d in direc:
        if direc[d] == -1: continue
        lst.append((direc[d], d))

    sorted_list = sorted(lst, key=lambda x: x[0])
    print(sorted_list)
    string = ""
    for (idx, val) in enumerate(sorted_list):
        if idx == 0: string = val[1]
        else:
            string = string + ' ' + val[1]
    return string
for uuid in _data:
    if uuid == 'tracklet_610':
        pass
    _direc = {}
    _color = {}
    _type = {}
    for nl in _data[uuid]['nl']:
        d = direction_process(direction[nl])
        if d not in _direc:
            _direc[d] = 0
        _direc[d] += 1
        c = subject[nl]['color']
        v = subject[nl]['vehicle']
        if c is not None:
            if c not in _color:
                _color[c] = 0
            _color[c] += 1
        if v is not None:
            if v not in _type:
                _type[v] = 0
            
            _type[v] += 1
    _output[uuid] = {
        "directions": _direc,
        "colors": _color,
        "type": _type
    }

print(_output)


with open('data/json/uuid/uuid_attributes_train.json', 'w') as outfile:
        json.dump(_output, outfile, indent=2)
    
