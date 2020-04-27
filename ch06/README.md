# Chapter 6: likelihood analysis of spatial capture-recapture models

These models use a grid approximation to marginalize over activity centers `s`
(as you would do for maximum likelihood estimation), but are Bayesian. 
This marginalization is not required in Stan (which can directly use `s`), but
these models show how it can be done.  


### Known N (SCR0)

See [scr0-known-n.R](scr0-known-n.R).

### Unknown N (SCR0)

1. **Binomial form**. See 
[scr0-unknown-n-binomial-form.R](scr0-unknown-n-binomial-form.R), which uses 
a binomial model for N (as in eq. 6.2.1).

2. **Data augmentation**. See [scr0-unknown-n-data-augmentation.R](scr0-unknown-n-data-augmentation.R).

### Wolverine case study 

1. **Binomial form**. See [scr0-wolverine.R](scr0-wolverine.R).

2. **Poisson integrated**. See [scr0-wolverine-poisson-integrated.R](scr0-wolverine-poisson-integrated.R).
Notice that even though we integrated `s` out of the model, we can still 
sample from `s` in the `generated quantities` block.
