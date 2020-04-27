# Chapter 9: alternative observation models

1. **Poisson observation model.** See 
[poisson-observations.R](poisson-observations.R) for a Poisson observation
example (note the use of `poisson_log_lpmf`).

2. **Categorical/multinoulli observation model.** See 
[categorical-observations.R](categorical-observations.R) for an example, and 
notice the use of `categorical_logit_lpmf`.

3. **Categorical multisession model (ovenbirds example).** See
[multisession-categorical.R](multisession-categorical.R) for an example. This
model gets somewhat complex, because there are a variable number of sampling
occasions in each year, and there are animals that are known to be dead on 
some sampling occasions. 

4. **Possum case study.** See [possum-example.R](possum-example.R).
