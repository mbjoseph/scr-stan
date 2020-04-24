library(scrbook)
library(secr)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Data processing code adapted from scrbook::SCRpossum()
data(possum)

X <- as.matrix(traps(possumCH))
X[, 1] <- X[, 1] - min(X[, 1])
X[, 2] <- X[, 2] - min(X[, 2])

y3d <- possumCH

n_obs <- dim(y3d)[1]
n_occasions <- dim(y3d)[2]
n_traps <- dim(y3d)[3]

y <- matrix(n_traps + 1, nrow = n_obs, ncol = n_occasions)
for (i in 1:n_obs) {
  for (j in 1:n_occasions) {
    if (any(y3d[i, j, ] == 1)) {
      y[i, j] <- which(y3d[i, j, ] == 1)
    }
  }
}

buff <- 100
xlim <- c(min(X[, 1]) - buff, max(X[, 1]) + buff)
ylim <- c(min(X[, 2]) - buff, max(X[, 2]) + buff)

M <- 300
yaug <- matrix(n_traps + 1, nrow = (M - n_obs), ncol = n_occasions)
Ycat <- rbind(y, yaug)
stopifnot(nrow(Ycat) == M)


stan_d <- list(
  M = M, 
  n_trap = n_traps, 
  n_occasion = n_occasions, 
  X = X, 
  y = Ycat, 
  xlim = xlim, 
  ylim = ylim
)

m_init <- stan_model("ch09/categorical-observations.stan")
m_fit <- sampling(m_init, data = stan_d)
traceplot(m_fit, pars = c("alpha0", "alpha1", "psi", "N"))
print(m_fit, pars = c("alpha0", "alpha1", "psi", "N"))
