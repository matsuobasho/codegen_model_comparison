### Test different code-generation LLMs against proprietary codebase

Code leverages AzureML to finetune existing models on codebase.
Make sure you are connected to your AzureML subscription and make sure that you
have already created a workspace.
We use the AzureML CLI syntax, so run the following in the command line.

Enter commands in the cli.

1. Configure default workspace and environment.  This allows you to not have to specify these arguments in subsequent calls.\
`az configure --defaults workspace=<ws_name> group=<resource_group_name>`

2. Create compute instance.  Specify the location and computes that you have access to under your Azure subscription in the compute yaml file, along with the compute name. \
`az ml compute create -f codegen_model_comparison/cloud/compute.yaml`
In pipeline.yaml, specify the compute you've created under the default_compute field.

2. Create environment.  This leverages the `environment.yaml` file to create a custom environment.\
`az ml environment create -f codegen_model_comparison/cloud/environment/environment.yaml`

3. Launch the job.\
`az ml job create -f codegen_model_comparison/cloud/pipeline.yaml`