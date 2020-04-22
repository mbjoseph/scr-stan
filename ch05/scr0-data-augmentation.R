library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data <- simSCR0(discard0 = TRUE, rnd = 2013)

n_aug <- 200
y <- rbind(data$Y, matrix(0, nrow = n_aug, ncol = ncol(data$Y)))

stan_d <- list(
  M = nrow(y),
  n_trap = nrow(data$traplocs), 
  n_occasion = data$K, 
  xlim = data$xlim, 
  ylim = data$ylim,
  X = data$traplocs, 
  y = y, 
  n_aug = n_aug
)

m_init <- stan_model("ch05/scr0-data-augmentation.stan")
m_fit <- sampling(m_init, data = stan_d)
traceplot(m_fit, pars = c("alpha1", "alpha0", "psi", "N"))
print(m_fit, pars = c("alpha1", "alpha0", "psi", "N"))
