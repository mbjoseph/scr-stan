library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

set.seed(2013)
parms <- list(N = 100, alpha0 = -.40, sigma = 0.5, alpha2 = 0)

# simulate some data
data <- simMnSCR(parms, K = 7, ssbuff = 2)

n_obs <- nrow(data$Ycat)
M <- 200
n_aug <- M - n_obs

Ycat <- rbind(data$Ycat, 
              matrix(nrow(data$X) + 1, 
                     nrow = n_aug, ncol = data$K)
              )


stan_d <- list(
  M = M, 
  n_trap = nrow(data$X), 
  n_occasion = data$K, 
  X = data$X, 
  y = Ycat, 
  xlim = data$xlim, 
  ylim = data$ylim
)

m_init <- stan_model("ch09/categorical-observations.stan")
m_fit <- sampling(m_init, data = stan_d)

# psi and N match up very nicely with results reported in book
# alpha0 and alpha1 are different, not sure why...
traceplot(m_fit, pars = c("alpha0", "alpha1", "psi", "N"))
pairs(m_fit, pars = c("alpha0", "alpha1", "psi"))
print(m_fit, pars = c("alpha0", "alpha1", "psi", "N"))
