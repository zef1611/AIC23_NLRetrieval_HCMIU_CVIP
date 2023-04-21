import json

queries = json.load(open("./data/json/original_data/test-queries.json"))
tracks = json.load(open("./data/json/original_data/test-tracks.json"))
inc_cam = ["c001", "c002", "c003", "c004", "c005", "c013", "c014", "c019", "c026", "c030", "c033", "c034", "c036", "c037", "c040"]

def get_camera(file_path):

    cam = file_path.split('/')[-3]
    return cam

def have_intersect(uuid):
    for nl in queries[uuid]['nl']:
        if 'intersection' in nl: 
            return True
    return False

def process(uuid_list):
    keep = []
    remove = []
    for uuid in uuid_list:
        uuid_cam = get_camera(tracks[uuid]['frames'][0])
        
        if uuid_cam in inc_cam:
            keep.append(uuid)
        
        else:
            remove.append(uuid)
    return keep + remove

def rerank_intersection(result_data):
    for query in result_data:
        if not have_intersect(query): 
            continue
        result_data[query] = process(result_data[query])
    return result_data