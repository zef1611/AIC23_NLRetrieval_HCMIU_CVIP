
retrieval='v2_standard_extend.json'
type='test-tracks-type.json'
color='test-tracks-color.json'
direction='test-tracks-direction-refinement.json'

python src/post_process/rerank/run_rerank.py --rank_dir './data/json/retrieval/'${retrieval} --type_dir './data/json/recognition/'${type} --color_dir './data/json/recognition/'${color} --direction_dir './data/json/recognition/'${direction} 