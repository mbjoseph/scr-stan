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
n_aug <- 400

y <- rbind(y_obs,
           matrix(0, nrow = n_aug, ncol = ncol(y_obs)))

buffer <- 5

stan_d <- list(M = n_obs + n_aug,
               n_aug = n_aug,
               n_obs = n_obs,
               n_trap = nrow(trap_coords), 
               n_occasion = dim(ymat)[3], 
               X = trap_coords, 
               y = y, 
               xlim = range(trap_coords[, 1]) + c(-1, 1) * buffer, 
               ylim = range(trap_coords[, 2]) + c(-1, 1) * buffer)

m_init <- stan_model("ch07/individual-heterogeneity-ranefs.stan")
m_fit <- sampling(m_init, data = stan_d)
traceplot(m_fit, pars = c("alpha1", "mu_alpha0", "sd_alpha0", "N"))
plot(m_fit, pars = "z_alpha0")


# Plot activity centers ---------------------------------------------------


plot_s <- function(individual, ...) {
  s_post <- rstan::extract(m_fit, pars = "s")$s
  s1_df <- as.data.frame(s_post[, individual, ])
  
  plot(trap_coords, xlim = stan_d$xlim, ylim = stan_d$ylim, 
       xaxt='n', yaxt='n', ann=FALSE)
  points(s1_df, col = scales::alpha("dodgerblue", 200 / nrow(s1_df)), pch = 19)
  points(trap_coords[y_obs[individual, ] > 0, ], pch = 19)
}

par(mfrow = c(2, 2), oma = c(0, 0, 0, 0), mar = c(.1, .1, .1, .1))
plot_s(1)
plot_s(2)
plot_s(3)
plot_s(4)
par(mfrow = c(1, 1))
