library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
# adapted from scrbook::`ch11secr-jags`

data(ch11simData)

# M is 100, but the true value is 30 - let's reduce M for efficiency
y <- ch11simData$ch.jags[1:60, ]

stan_d <- list(
  y = y, 
  canopy = drop(ch11simData$spcov.jags$CANHT), 
  M = nrow(y), 
  n_trap = nrow(ch11simData$traps), 
  n_grid = nrow(ch11simData$spcov.jags), 
  grid_pts = as.matrix(ch11simData$spcov.jags[, c("x", "y")]), 
  X = ch11simData$traps, 
  pixel_area = 25
)

init_fn <- function(chain_id) return(list(beta0 = runif(1, -9, -7)))

m_init <- stan_model("ch11/discrete-ipp-scr.stan")
m_fit <- sampling(m_init, data = stan_d, init = init_fn)
rstan::traceplot(m_fit, 
                 pars = c("beta0", "beta1", "loglam0", "alpha1", "log_en", 
                          "log_psi", "N"))
pairs(m_fit, pars = c("beta0", "beta1", "loglam0", "alpha1", "log_en", 
                      "log_psi", "N"))
