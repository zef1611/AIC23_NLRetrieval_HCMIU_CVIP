#!/bin/bash
if [ ! -d "logs" ]; then
    mkdir logs
fi

name='v2_standard_extend'
config='v2_standard_extend.yaml'

export CUDA_VISIBLE_DEVICES='1'
nohup python src/inference.py --config ${config} --name ${name} > logs/${name}_inference.log 2>&1 &