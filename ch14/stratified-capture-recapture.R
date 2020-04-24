library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


# adapted from scrbook::multisession_sim
set.seed(2013)
n_group <- 20
beta0 <- 3
beta1 <- 0.6
pr_detection <- 0.3
n_occasion <- 5
x <- rnorm(n_group)
lambda <- exp(beta0 + beta1 * x)
N <- rpois(n_group, lambda = lambda)
y <- NULL
for (g in 1:n_group) {
  if (N[g] > 0) 
    y <- c(y, rbinom(N[g], n_occasion, pr_detection))
}
g <- rep(1:n_group, N)
g <- g[y > 0]
y <- y[y > 0]
M <- 1200
g <- c(g, rep(0, M - length(g)))
y <- c(y, rep(0, M - length(y)))

stan_d <- list(M = length(y), 
               n_group = n_group, 
               obs_g = g, 
               n_occasion = n_occasion, 
               x = x, 
               y = y)

m_init <- stan_model("ch14/stratified-capture-recapture.stan")
m_fit <- sampling(m_init, data = stan_d)

traceplot(m_fit, pars = c("beta0", "beta1", "p_detect", "log_psi", "N", "Ng"))
pairs(m_fit, pars = c("beta0", "beta1", "p_detect", "log_psi"))
