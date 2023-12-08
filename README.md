### Test different code-generation LLMs against proprietary codebase

Code leverages AzureML to finetune existing models on codebase.
Make sure you are connected to your AzureML subscription.
We use the AzureML CLI syntax, so run the following in the command line.

1. Open up the shell script referenced below and change the name of the resource
group to yours.  Run the shell script from your command line.

`sh codegen_model_comparison/run_pipeline.sh`

2. After it finishes, run:

`az ml job list -r 1`

3. Then run the next line with the job name from the previous result:
`az ml job download --all -n <job name>`

----------------
Architecture:
We are interested in 3 things:
1. Establish baseline code model performance on test functions
2. Finetune code models with our own functions
3. Examine finetuned model performance on test functions

Though these are 3 distinct steps, we have to be mindful of using computational
resources efficiently.  Since steps 1 and 3 require the same models to be loaded,
we combine them into 1 Azure component.  So we have essentially 2 components that
wan reuse for different code-generating models that we want to run.

-------------------
Thoughts
- In theory, highly descriptive docstrings for functions that have a set
structure to them should lend themselves well to language modeling
- OTOH, there are lots of outside concepts that the LM doesn't know to incorporate
(in this case physics), and there are also interpdendencies, which if not spelled
out explicitly, the LM won't know what to do with
- We want the model to be robust to imperfections
- I began the project aspiring to incorporate hyperparameter tuning, but decided
to simplify to just static parameters as a first step


