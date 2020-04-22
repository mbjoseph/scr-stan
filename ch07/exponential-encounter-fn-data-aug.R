library(scrbook)
library(sf)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data("beardata")
ymat <- beardata$bearArray
trap_coords <- beardata$trapmat

y_obs <- apply(ymat, 1:2, sum)

n_obs <- nrow(y_obs)
n_aug <- 650

y <- rbind(y_obs,
           matrix(0, nrow = n_aug, ncol = ncol(y_obs)))

buffer <- 2

stan_d <- list(M = n_obs + n_aug,
               n_aug = n_aug,
               n_trap = nrow(trap_coords), 
               n_occasion = dim(ymat)[3], 
               X = trap_coords, 
               y = y, 
               xlim = range(trap_coords[, 1]) + c(-1, 1) * buffer, 
               ylim = range(trap_coords[, 2]) + c(-1, 1) * buffer)

m_init <- stan_model("ch07/exponential-encounter-fn-s-data-aug.stan")
m_fit <- sampling(m_init, data = stan_d)
traceplot(m_fit, pars = c("alpha1", "alpha0", "N"))
