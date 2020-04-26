library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# adapted from scrbook::uniform_search
N <- 100
nyear <- 4
Sx <- Sy <- matrix(NA, nrow = N, ncol = nyear)
sigma.move <- 0.25
xlim <- c(0, 16)
ylim <- c(0, 16)
Sx[, 1] <- runif(N, xlim[1], xlim[2])
Sy[, 1] <- runif(N, ylim[1], ylim[2])
for (t in 2:nyear) {
  Sx[, t] <- rnorm(N, Sx[, t - 1], sigma.move)
  Sy[, t] <- rnorm(N, Sy[, t - 1], sigma.move)
}
f <- rep(NA, N)
Y <- matrix(0, nrow = N, ncol = nyear)
for (i in 1:N) {
  for (t in 1:nyear) {
    if (Sx[i, t] > 3 & Sx[i, t] < 13 & Sy[i, t] > 3 & 
        Sy[i, t] < 13) 
      Y[i, t] <- rbinom(1, 1, 0.5)
  }
  if (sum(Y[i, ]) > 0) 
    f[i] <- min((1:nyear)[Y[i, ] == 1][1])
}
Y <- Y[!is.na(f), ]
Sx <- Sx[!is.na(f), ]
Sy <- Sy[!is.na(f), ]
Sx[Y == 0] <- NA
Sy[Y == 0] <- NA
M <- 200
Y <- rbind(Y, matrix(0, nrow = (M - nrow(Y)), ncol = nyear))
Sx <- rbind(Sx, matrix(NA, nrow = (M - nrow(Sx)), ncol = nyear))
Sy <- rbind(Sy, matrix(NA, nrow = (M - nrow(Sy)), ncol = nyear))
G <- array(NA, dim = c(M, nyear, 2))
G[, , 1] <- Sx
G[, , 2] <- Sy

i_mat <- matrix(rep(1:M, times = nyear), nrow = M)
k_mat <- matrix(rep(1:nyear, each = M), nrow = M)



# A continuous approximation of the step function -------------------------
# The "true" generative model and the JAGS model use step functions to 
# determine whether individuals have a chance of being detected. Stan needs
# the log probability to be differentiable, so we'll approximate the 
# discontinuous step function with a differentiable continuous function. 
phi <- 20

logistic_fn <- function(x, steepness, midpoint) {
  1 / (1 + exp(-steepness * (x - midpoint)))
}
xseq <- seq(xlim[1], xlim[2], .1)
f1 <- logistic_fn(xseq, phi, 3)
f2 <- 1 - logistic_fn(xseq, phi, 13)
plot(xseq, f1, type = "l", ylab = "", xlab = "x1")
lines(xseq, f2)
abline(v = c(3, 13), lty = 2)
# our approximation is the product of two known logistic functions, 
# and approaches a step function as phi -> infinity.
lines(xseq, f1 * f2, col = "red", lty = 3, lwd = 5)




# Fit the model in Stan ---------------------------------------------------
stan_d <- list(
  M = M, 
  n_occasion = nyear, 
  y = Y, 
  xlim = xlim, 
  ylim = ylim, 
  n_detections = sum(Y), 
  ux_obs = Sx[Y == 1], 
  uy_obs = Sy[Y == 1], 
  i_obs = i_mat[Y == 1], 
  k_obs = k_mat[Y == 1], 
  n_unk = sum(Y == 0),
  i_unk = i_mat[Y == 0], 
  k_unk = k_mat[Y == 0], 
  phi = phi, 
  search_x = c(3, 13), 
  search_y = c(3, 13)
)

# compilation raises warnings about potential non-linear transformations
# but these can be safely ignored (we didn't nonlinearly transform u)
m_init <- stan_model("ch15/rw-uniform-search.stan")
m_fit <- sampling(m_init, data = stan_d)
# note that we will always get R-hat warnings, because some transformed params
# are observed without error (the location data), and those Rhats will be NA. 
# Defaults result in transitions that exceed maximum treedepth.
# May also get low BFMI warnings - running for more iterations might help.

traceplot(m_fit, pars = c("p0", "psi", "sigma_move", "N"))
traceplot(m_fit, pars = paste0("ux_unk[", 1:40, "]"))
print(m_fit, pars = c("p0", "psi", "sigma_move", "N"))
pairs(m_fit, pars = c("p0", "psi", "sigma_move", "N", "lp__"))
