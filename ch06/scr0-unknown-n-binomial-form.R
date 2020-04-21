library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data <- simSCR0(discard0 = TRUE, rnd = 2013, N = 30)

# setup a grid to approximate the marginalization over space
# smaller delta values --> better approximation
delta <- .3
x1_grid <- seq(data$xlim[1], data$xlim[2], by = delta)
x2_grid <- seq(data$ylim[1], data$ylim[2], by = delta)
grid_pts <- expand.grid(x1 = x1_grid, x2 = x2_grid) 


stan_d <- list(
  n_nonzero_histories = nrow(data$Y),
  n_trap = nrow(data$traplocs), 
  n_occasion = data$K, 
  n_grid = nrow(grid_pts), 
  grid_pts = as.matrix(grid_pts),
  X = data$traplocs, 
  y = data$Y, 
  n0_prior_scale = 10
)

m_init <- stan_model("ch06/scr0-unknown-n-binomial-form.stan")
m_fit <- sampling(m_init,  data = stan_d)
traceplot(m_fit, pars = c("alpha1", "alpha0", "N"))
