library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

data <- simSCR0(discard0 = FALSE, rnd = 2013, N = 30)

buff_dist <- 3
buff_adj <- c(-1, 1) * buff_dist

stan_d <- list(
  n_individual = nrow(data$Y),
  n_trap = nrow(data$traplocs), 
  n_occasion = data$K, 
  xlim = data$xlim + buff_adj, 
  ylim = data$ylim + buff_adj,
  X = data$traplocs, 
  y = data$Y, 
  n_zero_histories = sum(rowSums(data$Y) == 0)
)

m_init <- stan_model("ch05/scr0.stan")
m_fit <- sampling(m_init,  data = stan_d)
traceplot(m_fit, pars = c("alpha1", "alpha0"))
