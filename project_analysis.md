## Introduction
One Friday afternoon, while planning the software development work for the
following week, a thought crossed my mind, "Wouldn't it be nice if I could
issue a set of instructions about the intended feature and have the machine
take at least a first pass at writing it for me."

Large language models have gotten a lot of attention in 2023 (from hereon just referred
to as LMs).  So the idea was to see how well these LMs, finetuned on
our company's codebase, perform on a much more simplifed task.

First to get it out the way, I am of course familiar of course with Github Copilot.
But Copilot is paid, and I would also like control over the internals
of the LMs as opposed to just having a black box.  Relying on a copmany,
as the saga with OpenAI in the past couple of weeks has reminded the world,
can give you a false sense of confidence.

Designing a system to create interrelated blocks of code that integrate into
a functioning codebase in response to a user command is a very difficult
experiment, so I limited the scope to something much more
managebale.  Namely, generating detailed code from Python function documentation
(from hereon referred to as docstrings).


## Data
In our codebase, we strive to adhere to standards for both our docstrings
and the Python functions.  Every docstring has at the minimum the same sections with a
description of what it does, inputs and outputs.  Additionally, we intentionally
included layperson-friendly explanations of the engineering and solar concepts pertinent to a
function.

Although the code itself doesn't have the same rigid structure, we strive for
a convention for variable names, a certain coding style (Pandas/numpy heavy
vectorization, writing for humans), and adhere to the Google coding standard.

So we can use the docstrings, which in essence describe what the function
does, to generate the function itself.

## The questions
So I wanted to know:
* What LMs can we test?
* How much do they improve if we finetuned them as oppoosed to just using
    them out of the box?
* How good (or bad) is the code they generate?  Does it even run?

## Models
We decided on the following 3:
* SalesForce Codegen 350M
    * 350 million parameters
    * Trained on 71.8B Python tokens
* Decicoder 1b
    * 1 billion parameters
    * Trained on the Python, Java, and Javascript subsets of Starcoder Training Dataset
* CodeLlama 7b
    * 7 billion parameters

Model-training details would go here.

## Baseline predictions

I chose 3 functions that represent the range of complexity within our code.
I will share 2 of them (the 3rd one has our secret sauce for quantifying
dealing with uncertainty in energy losses).

There are adjustable parameters with the generation phase that one can tweak
as well.  Here, we used !!! talk about what was ued.

### **Function 1:**
Calculate pv efficiency loss.  !!! Provide a link to the function here

#### Prediction Codegen:
> *** # Calculating LFPs per day ######################################################## ---------------------------   ----------------------      ------------------------------       ------------------------------------                                -------------------------------------          -------------------------------------------------------------         ------------------------------------------------------------------------               --------------------------------------------------------                           ----------           =================================================                      =======================================================================                         **************************************************                       __________________________________________________________                        ___________________________________________________________________________________                            ...................................................................................................................                              //////////////////////////////////////                    ..........................                         ....                   .................                 ..................                  .....               .......                           ......            .............           .#              ##                 ######## ##... ###..................... #####
            | |	||		 ||			||||																								│								 │							 ┌─╸̅͘═´¯`·‿»¬¶•Àº«¨®ƒœ№●†Â©™Ã±ôˆ¢˜ħø∙ß€£$&*()+–—→½âêîûëïöüçãàèẽðéóúñæåäíǎoʼɪțᵉδύπαρμος εκτινελωσγ χác ở Điūr ánātōu žaškējībén kai mikrofon takus ved nyauzumot luktiem daraje og sveitam udavimoseten bij eksplorativni tehte görjaan jaunomis zapsojučia işlemeji vektoriye hilvasojamega arabievi yerdezaqta qaytoje niurraki razkojoğunut koordinaatsizna farkında birini tanlangacaks iradigineci kelimeyi alma sahiptama verdi!\xa0", \'''PV Systs\'', 'Location': {}, \'Uncertainties:\'. ', '']}')***

#### Prediction Decicoder:
>   \# Calculate start effecitve levels assuming constant relative error
>     self._calc()
>
>    \#    def _getlocdata():
>         return locdf['location'].values[startidx:]

### **Function 2:**
Get distance between 2 geographic coordinates.  !!! Provide a link here.

#### Prediction Codegen:
> ***# Get distances from each point to all other points on this line segment: (1) find closest neighbor; 2). Find nearest neighbour for every index that is within range [0..360]!!! !!!!!!!!!!!!!!!???????????? = 1/2 * 360^n + n where 0 <= i < len(_indexes)-i - 3 # TODO check if we can use np instead or just using pandarallel's linear interpolation function here... maybe not as it seems like there are many ways but I think its ok because they're very similar anyways... :)  !!!!!!!!?????????     = https://stackoverflow-questions@python3k/?qid=(RU5TJQKLZYHXWVFTVA4B7EI6M8DG9C&tb=/rjmhgwvfzcvs%40gmailcomposermailboxapplicationservicesystemsolutionsandemailaddressoftheuserwithaverythingthatisnotinhereofthesystemsoverwritenotthisoneforyou!)
***

#### Prediction Decicoder:
> *** # Convert to radian values for easier math calculations later on:
    lon = np***

So far, this is just good for a laugh.  Of the 3 output, my favorite one is Codegen's
prediction of function 1, which first shows the model's affinity for dashes, equal
signs and asterisks, proceds to a random assortment of Greek, what looks to be some central
European language (maybe Czech?), and concludes with 3 solar-related terms.

So we at least know it's incorporating some information from the docstring.


## Finetuned predictions
Ok, now let's examine the predictions on the finetuned dataset.

### Function 1
#### Codegen
<blockquote> name = 'lf_pv_eff'
logging.info('Calculating %s', name)
lpb, upb, asymmetry_factor, pv_eff_unc, pv_lid_coef, pv_degrad_coef, pv_mql_coef = user_settings.ilf_parameters[
    name]

\# uncertainty calculation
pv_lid_var = self.sdm_est[self.var_names.index('PV Meter, Real Power')]
pv_lid_unc = self.sdm_est_unc_half[self.var_names.index('PV Meter, Real Power')]
pv_degrad_var = self.sdm_est[self.var_names.index('PV Meter, Real Power')]
pv_degrad_unc = self.sdm_est_unc_half[self.var_names.index('PV Meter, Real Power')]
pv_mql_var = self.sdm_est[self.var_names.index('PV Meter, Real Power')]
pv_mql_unc = self.sdm_est_unc_half[self.var_names.index('PV Meter, Real Power')]

\# calculate lower and upper bounds
lpb = lpb * pv_eff_unc / 100
lpb = pd.DataFrame(np.clip(lpb, a_min=0, a_max=None),
                    index=self.sdm_data.index,
                    columns=self.sdm_data.columns)
lpb.name = 'lf_bnds'

upb = upb * pv_eff_unc / 100
upb = pd.DataFrame(np.clip(upb, a_min=0, a_max=None),
                    index=self.sdm_data.index,
                    columns=self.sdm_data.columns)
upb.name = 'lf_bnds'

\# calculate lower and upper bounds for degradation
pv_degrad_var_lower = pv_degrad_var - pv_degrad_coef * lpb.sum()
pv_degrad_var_upper = pv_degrad_var + pv_degrad_coef * upb.sum()

\# calculate lower and upper bounds for module quality loss
pv_mql_var_lower = pv_mql_var - pv_mql_coef * lpb.sum()
pv_mql_var_upper = pv_mql_var + pv_mql_coef * upb.sum()

\# create dataframe with only the relevant columns
df = pd.DataFrame(np.hstack((pv_lid_var, pv_lid_unc, pv_degrad_var,
                            pv_degrad_unc, pv_mql_var, pv_mql_unc)),
                    columns=self.sdm_data.columns)

\# add asymmetry factor
df['lf_bnds'] = df['lf_bnds'] * (1 + asymmetry_factor)

\# add lower and upper bounds to dataframe
df.loc[:, 'lf_bnds'] = pd.concat([df['lf_bnds'], lf_bnds], axis=1)
df.loc[:, 'lf_bnds'] = pd.concat([df['lf_bnds'], lf_bnds], axis=1)

return df
</blockquote>

#### Decicoder

### Function 2
#### Codegen

#### Decicoder

## Metrics

## Drawbacks
    Even if the results are good, the best results with this setup
    can only write a function.  That's a far cry from being able to generate
    a whole set of code that successfully incorporates a new feature in the
    codebase.  So what we are ignoring is class definitions, how classes relate
    to each other.