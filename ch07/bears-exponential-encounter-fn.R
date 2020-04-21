library(scrbook)
library(sf)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data("beardata")
ymat <- beardata$bearArray
trap_coords <- beardata$trapmat

buffer <- 3

trap_sf <- st_as_sf(trap_coords, coords = c("V1", "V2"))
trap_buffer <- st_buffer(trap_sf, buffer)
grid <- st_make_grid(trap_buffer, cellsize = 1, what = "centers")

plot(grid, cex = .5, col = "grey")
plot(trap_sf, add = TRUE, pch = 19)

y_obs <- apply(ymat, 1:2, sum)

stan_d <- list(n_nonzero_histories = nrow(y_obs), 
               n_trap = nrow(trap_coords), 
               n_occasion = dim(ymat)[3], 
               n_grid = length(grid), 
               grid_pts = st_coordinates(grid), 
               X = trap_coords, 
               y = y_obs, 
               n0scale = 200)

m_init <- stan_model("ch07/bears-exponential-encounter-fn.stan")
m_fit <- sampling(m_init, data = stan_d, iter = 200)
traceplot(m_fit, pars = c("alpha1", "alpha0", "N"))


# Plot activity centers ---------------------------------------------------


plot_s <- function(individual) {
  s_post <- rstan::extract(m_fit, pars = "s")$s
  s1_df <- as.data.frame(s_post[, individual, ])
  s1_sf <- st_as_sf(s1_df, coords = c("V1", "V2"))
  
  plot(grid, cex = .2, col = "grey")
  plot(trap_sf, add = TRUE)
  plot(trap_sf[stan_d$y[individual, ] > 0, ], pch = 19, add = TRUE)
  plot(s1_sf, col = scales::alpha("red", .01), add = TRUE, pch = 19)
}

plot_s(1)
plot_s(2)
plot_s(3)
plot_s(4)
