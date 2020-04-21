library(scrbook)
library(rstan)
library(sf)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)



# Munge data --------------------------------------------------------------

data(wolverine)

traps <- wolverine$wtraps
traplocs <- traps[, c("Easting", "Northing")] / 10000
K.wolv <- apply(traps[, 4:ncol(traps)], 1, sum)

y3d <- SCR23darray(wolverine$wcaps, traps)
y2d <- apply(y3d, c(1, 2), sum)

# setup a grid to approximate the marginalization over space
# smaller delta values --> better approximation
delta <- .5
buffer <- 3
x1_grid <- seq(min(traplocs$Easting) - buffer, 
               max(traplocs$Easting) + buffer, 
               by = delta)
x2_grid <- seq(min(traplocs$Northing) - buffer, 
               max(traplocs$Northing) + buffer, 
               by = delta)
grid_pts <- expand.grid(Easting = x1_grid, Northing = x2_grid) 



# filter points based on distance to traps ---------------
grid_sf <- st_as_sf(grid_pts, coords = c("Easting", "Northing"))
trap_sf <- st_as_sf(traplocs, coords = c("Easting", "Northing"))
trap_buffer <- st_buffer(trap_sf, buffer)

# this horrorshow removes grid points outside trap buffers 
grid_sf <- grid_sf[unlist(lapply(st_intersects(grid_sf, trap_buffer), length)) > 0, ]

plot(grid_sf, pch = 4, col = "red", cex = .3)
plot(trap_sf, add = TRUE)



# Use Stan ----------------------------------------------------------------

stan_d <- list(
  n_nonzero_histories = nrow(y2d),
  n_trap = nrow(traplocs), 
  n_occasion = K.wolv, 
  n_grid = nrow(grid_sf), 
  grid_pts = st_coordinates(grid_sf),
  X = traplocs, 
  y = y2d, 
  n0_prior_scale = 10
)

m_init <- stan_model("ch06/scr0-wolverine-poisson-integrated.stan")
m_fit <- sampling(m_init,  data = stan_d)
traceplot(m_fit, pars = c("alpha1", "alpha0", "N"))


# Visualize posteriors for activity centers s -----------------------------

plot_s <- function(individual) {
  s_post <- rstan::extract(m_fit, pars = "s")$s
  s1_df <- as.data.frame(s_post[, individual, ])
  s1_sf <- st_as_sf(s1_df, coords = c("V1", "V2"))
  
  plot(grid_sf, cex = .2, col = "grey")
  plot(trap_sf, add = TRUE)
  plot(trap_sf[y2d[individual, ] > 0, ], pch = 19, add = TRUE)
  plot(s1_sf, col = scales::alpha("red", .05), add = TRUE, pch = 19)
}

plot_s(1)
plot_s(2)
plot_s(12)
