# Chapter 7: variation in encounter probability

These models introduce alternative encounter functions, individual-, time-, and 
sex-varying encounter models. 

### Exponential encounter functions

1. The [exponential-encounter-fn.R](exponential-encounter-fn.R) demonstrates
the use of an exponential encounter function + integrated likelihood, 
marginalizing over `s`. 

2. See [exponential-encounter-fn-data-aug.R](exponential-encounter-fn-data-aug.R)
for a model that uses data augmentation, keeps `s` in the joint probability 
(instead of marginalizing), and uses an exponential encounter function. 

### Heterogeneity in encounter models

1. **Individual-level**. See
[individual-heterogeneity-ranefs.R](individual-heterogeneity-ranefs.R) 
for a model with individual-level normal random effects. Notice that for 
efficiency, this model uses a non-centered parameterization. For some background
see: https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html

2. **Time-varying**. See [time-varying-p0.R](time-varying-p0.R) for a model
where `p0` varies by occasion. This also uses a non-centered parameterization.

3. **Sex-varying**. See 
[sex-as-individual-covariate.R](sex-as-individual-covariate.R) for a model
that allows encounter probability to vary with sex. The challenge here is that 
sex is only observed for a subset of individuals. The likelihood for unobserved
individuals marginalizes over sex (a latent individual-level covariate).
