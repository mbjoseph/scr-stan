library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data <- simPoissonSCR(discard0 = TRUE, rnd = 2013)

n_obs <- nrow(data$Y)
M <- 200
n_aug <- M - n_obs

y <- rbind(data$Y, matrix(0, nrow = n_aug, ncol = ncol(data$Y)))

stan_d <- list(
  M = M, 
  n_aug = n_aug, 
  n_obs = n_obs, 
  n_trap = ncol(y), 
  n_occasion = data$K, 
  X = data$traplocs, 
  y = y, 
  xlim = data$xlim, 
  ylim = data$ylim
)

m_init <- stan_model("ch09/poisson-observations.stan")
m_fit <- sampling(m_init, data = stan_d)
traceplot(m_fit, pars = c("alpha0", "alpha1", "N"))
print(m_fit, pars = c("alpha0", "alpha1", "N"))
