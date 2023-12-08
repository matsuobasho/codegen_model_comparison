#!/bin/bash

az ml workspace create -g aus-rg -n dec-ws
az configure --defaults group=aus-rg workspace=dec-ws
az ml compute create -f codegen_model_comparison/cloud/compute.yaml
az ml environment create -f codegen_model_comparison/cloud/environment/environment.yaml

az ml data create --name functions --version 1 --path codegen_model_comparison/data/dataset_hf_train_val.pkl --type uri_file
az ml job create -f codegen_model_comparison/cloud/pipeline.yaml

# After the job finishes
# az ml job list -r 1

# # input the child job name from the previous step
# az ml job download --all -n <job name>