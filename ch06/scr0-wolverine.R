library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data(wolverine)

traps <- wolverine$wtraps
traplocs <- traps[, c("Easting", "Northing")] / 10000
K.wolv <- apply(traps[, 4:ncol(traps)], 1, sum)

y3d <- SCR23darray(wolverine$wcaps, traps)
y2d <- apply(y3d, c(1, 2), sum)

# setup a grid to approximate the marginalization over space
# smaller delta values --> better approximation
delta <- .5
buffer <- 1
x1_grid <- seq(min(traplocs$Easting) - buffer, 
               max(traplocs$Easting) + buffer, 
               by = delta)
x2_grid <- seq(min(traplocs$Northing) - buffer, 
               max(traplocs$Northing) + buffer, 
               by = delta)
grid_pts <- expand.grid(x1 = x1_grid, x2 = x2_grid) 

plot(grid_pts, cex = .5, col = "red", pch = 3)
points(traplocs)

stan_d <- list(
  n_nonzero_histories = nrow(y2d),
  n_trap = nrow(traplocs), 
  n_occasion = K.wolv, 
  n_grid = nrow(grid_pts), 
  grid_pts = as.matrix(grid_pts),
  X = traplocs, 
  y = y2d, 
  n0_prior_scale = 10
)

m_init <- stan_model("ch06/scr0-wolverine.stan")
m_fit <- sampling(m_init,  data = stan_d)
traceplot(m_fit, pars = c("alpha1", "alpha0", "N"))
