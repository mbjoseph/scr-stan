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


# Visualize some activity centers --------------------------------
library(ggplot2)
library(patchwork)

post <- rstan::extract(m_fit)

plot_center <- function(ind) {
  ggplot(trap_coords, aes(V1, V2)) + 
    theme_void() + 
    stat_density_2d(aes(fill = after_stat(level)), geom = "polygon", 
                    data = as.data.frame(post$s[, ind, ]), 
                    color = "white", size = .05) +
    scale_fill_gradient(low = "white", high = "#b2001d") +
    theme(legend.position = "none") + 
    geom_point(alpha = .4, shape = 3, size = .8) + 
    geom_point(data = trap_coords[y[ind, ] > 0, ]) + 
    xlim(stan_d$xlim) + 
    ylim(stan_d$ylim) + 
    coord_equal()
}
p <- wrap_plots(lapply(c(40, 3, 46), plot_center), nrow = 1)
p
ggsave("activity-center.png", width = 8, height = 2, dpi = 450)
