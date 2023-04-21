import json
import rerank_attr
import intersection
import argparse

parser = argparse.ArgumentParser('rerank')
parser.add_argument('--rank_dir', type=str, help='input directory of rank file')
parser.add_argument('--type_dir', type=str, help='input directory of type prediction file')
parser.add_argument('--color_dir', type=str, help='input directory of color prediction file')
parser.add_argument('--direction_dir', type=str, help='input directory of direction prediction file')
args = parser.parse_args()

rank_dir = args.rank_dir
type_dir = args.type_dir
colors_dir = args.color_dir
directions_dir = args.direction_dir

cur_rank = json.load(open(rank_dir))
type_dict = json.load(open(type_dir))
colors_dict = json.load(open(colors_dir))
directions_dir = json.load(open(directions_dir))

final_rank = intersection.rerank_intersection(rerank_attr.Rerank(cur_rank, type_dict, colors_dict, directions_dir))

json.dump(final_rank, open('./data/json/retrieval/final_rank_submission.json', 'w'), indent=4)