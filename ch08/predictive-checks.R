library(scrbook)
library(rstan)
library(sf)
library(dplyr)
library(ggplot2)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


# Posterior checks for spatial randomness -----------------
# data processing code from scrbook::wolvSCR0ms()
data("wolverine")
traps <- wolverine$wtraps
y3d <- SCR23darray(wolverine$wcaps, wolverine$wtraps)

traplocs <- as.matrix(traps[, 2:3])
mingridx <- min(traplocs[, 1])
mingridy <- min(traplocs[, 2])
traplocs[, 1] <- traplocs[, 1] - min(traplocs[, 1])
traplocs[, 2] <- traplocs[, 2] - min(traplocs[, 2])
traplocs <- traplocs/10000
ntraps <- nrow(traplocs)

buffer <- 2
Xl <- min(traplocs[, 1] - buffer)
Xu <- max(traplocs[, 1] + buffer)
Yl <- min(traplocs[, 2] - buffer)
Yu <- max(traplocs[, 2] + buffer)

delta <- .1
ppc_grid <- expand.grid(x1 = seq(Xl, Xu, by = delta), 
                        x2 = seq(Yl, Yu, by = delta))


st_bbox_by_feature = function(x) {
  # stolen from https://github.com/r-spatial/sf/issues/1179
  x = st_geometry(x)
  f <- function(y) st_as_sfc(st_bbox(y))
  do.call("c", lapply(x, f))
}

grid_sf <- st_as_sf(ppc_grid, coords = c("x1", "x2")) %>%
  st_buffer(delta / 2) %>%
  st_bbox_by_feature()

ppc_grid_df <- grid_sf %>%
  lapply(st_coordinates) %>%
  lapply(as_tibble) %>%
  bind_rows(.id = "pixel") %>%
  mutate(pixel = as.numeric(pixel)) %>%
  group_by(pixel) %>%
  summarize(x1_min = min(X), 
            x1_max = max(X), 
            x2_min = min(Y), 
            x2_max = max(Y))

area <- (Xu - Xl) * (Yu - Yl)/10

M <- 200
n_obs <- dim(y3d)[1]
n_aug <- M - n_obs

MASK <- traps[, 4:ncol(traps)]
Dmat <- as.matrix(dist(traplocs))
K <- dim(y3d)[3]
newy <- array(0, dim = c(M, ntraps, K))
for (j in 1:n_obs) {
  newy[j, 1:ntraps, 1:K] <- y3d[j, 1:ntraps, 1:K]
}

K <- apply(MASK, 1, sum)
y <- apply(newy, c(1, 2), sum)



# Fit both models in Stan -------------------------------------------------


stan_d <- list(M = M,
               n_aug = n_aug,
               n_obs = n_obs,
               n_trap = ntraps, 
               n_occasion = K, 
               X = traplocs, 
               y = y, 
               xlim = c(Xl, Xu), 
               ylim = c(Yl, Yu), 
               n_cell = nrow(ppc_grid_df), 
               cell_xmin = ppc_grid_df$x1_min, 
               cell_xmax = ppc_grid_df$x1_max, 
               cell_ymin = ppc_grid_df$x2_min, 
               cell_ymax = ppc_grid_df$x2_max)

m_init <- stan_model("ch08/predictive-checks.stan")
m_fit <- sampling(m_init, data = stan_d)
traceplot(m_fit, pars = c("alpha0", "alpha1", "psi", "N"))


# Compute Bayesian p-values -----------------------------------------------

get_pvalue <- function(fit, param_prefix) {
  stat_obs <- rstan::extract(fit, pars = paste0(param_prefix, "_obs"))
  stat_sim <- rstan::extract(fit, pars = paste0(param_prefix, "_sim"))
  mean(unlist(stat_sim) > unlist(stat_obs))
}

# spatial randomness in the underlying point process
get_pvalue(m_fit, "freeman_tukey")
get_pvalue(m_fit, "index_dispersion")

# observation model checks
get_pvalue(m_fit, "T1")
get_pvalue(m_fit, "T2")
get_pvalue(m_fit, "T3")



# Visualize the spatial distribution of activity centers ------------------
Ng_post <- rstan::extract(m_fit, pars = "Ng")$Ng

grid_sf %>%
  st_as_sf %>%
  mutate(Ng = apply(Ng_post, 2, mean)) %>%
  ggplot() + 
  geom_sf(aes(fill = Ng), color = NA) + 
  scale_fill_viridis_c("N(g)") + 
  theme_minimal() + 
  theme(axis.text = element_blank(), 
        axis.title = element_blank()) + 
  annotate(x = traplocs[, 1], y = traplocs[, 2], geom = "point", 
           color = "white", shape = 1, alpha = .5)
