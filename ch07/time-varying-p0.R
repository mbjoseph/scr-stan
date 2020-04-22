library(scrbook)
library(sf)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data("beardata")
ymat <- beardata$bearArray
trap_coords <- beardata$trapmat

n_obs <- dim(ymat)[1]
n_trap <- dim(ymat)[2]
n_occasion <- dim(ymat)[3]
n_aug <- 150

# initialize augmented y array to zero
y <- array(0, dim = c(n_obs + n_aug, n_trap, n_occasion))
for (i in 1:n_obs) {
  # fill in observed (potentially nonzero) elements
  y[i, , ] <- ymat[i, , ]
}

buffer <- 5

stan_d <- list(M = n_obs + n_aug,
               n_aug = n_aug,
               n_trap = nrow(trap_coords), 
               n_occasion = dim(ymat)[3], 
               X = trap_coords, 
               y = y, 
               xlim = range(trap_coords[, 1]) + c(-1, 1) * buffer, 
               ylim = range(trap_coords[, 2]) + c(-1, 1) * buffer)

m_init <- stan_model("ch07/time-varying-p0.stan")
m_fit <- sampling(m_init, data = stan_d)
traceplot(m_fit, pars = c("alpha1", "alpha0", "N", "psi"))


# Plot activity centers ---------------------------------------------------


plot_s <- function(individual, ...) {
  s_post <- rstan::extract(m_fit, pars = "s")$s
  s1_df <- as.data.frame(s_post[, individual, ])
  trap_counts <- apply(y, 1:2, sum)[individual, ]
  
  plot(trap_coords, xlim = stan_d$xlim, ylim = stan_d$ylim, 
       xaxt='n', yaxt='n', ann=FALSE)
  points(s1_df, col = scales::alpha("dodgerblue", 200 / nrow(s1_df)), pch = 19)
  points(trap_coords[trap_counts > 0, ], pch = 19)
}

par(mfrow = c(2, 2), oma = c(0, 0, 0, 0), mar = c(.1, .1, .1, .1))
plot_s(1)
plot_s(2)
plot_s(3)
plot_s(4)
par(mfrow = c(1, 1))

