library(scrbook)
library(rstan)
library(rgeos)
library(sp)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Adapted from ?scrbook::snakeline
basex <- c(0, 0, 1, 1, 0)
basey <- c(0, 1, 1, 0, 0)
base <- cbind(basex, basey)

p1 <- cbind(basex, basey + 3)
p2 <- cbind(basex + 1, basey + 3)
p3 <- cbind(basex + 1, basey + 2)
p4 <- cbind(basex + 2, basey + 2)
p5 <- cbind(basex + 1, basey + 1)
p6 <- cbind(basex + 2, basey + 1)
p7 <- cbind(basex + 2, basey)
pp <- rbind(p1, p2, p3, p4, p5, p6, p7, base)

line1 <- read.csv("ch15/line.csv")

points <- SpatialPoints(line1)
sLine <- Line(points)
regpoints <- spsample(sLine, 100, type = "regular")
plot(line1, type = "l")
points(points, col = "grey")
points(regpoints, col = "red", pch = 20, lwd = 2)

perbox <- 4
N <- 30 * perbox
xlim <- c(-1, 4)
ylim <- c(-1, 5)
set.seed(2014)
sx <- runif(N, xlim[1], xlim[2])
sy <- runif(N, ylim[1], ylim[2])
points(sx, sy, pch = 20, col = "red")
sigma.move <- 0.35
sigma <- 0.4
alpha0 <- 0.8
alpha1 <- 1/(2 * (sigma^2))
X <- regpoints@coords
J <- nrow(X)
K <- 10
U <- array(NA, dim = c(N, K, 2))
y <- pmat <- matrix(NA, nrow = N, ncol = K)
for (i in 1:N) {
  for (k in 1:K) {
    U[i, k, ] <- c(rnorm(1, sx[i], sigma.move), 
                   rnorm(1, sy[i], sigma.move))
    dvec <- sqrt((U[i, k, 1] - X[, 1])^2 + (U[i, k, 2] - 
                                              X[, 2])^2)
    loghaz <- alpha0 - alpha1 * dvec * dvec
    H <- sum(exp(loghaz))
    pmat[i, k] <- 1 - exp(-H)
    y[i, k] <- rbinom(1, 1, pmat[i, k])
  }
}
Ux <- U[, , 1]
Uy <- U[, , 2]
Ux[y == 0] <- NA
Uy[y == 0] <- NA
points(Ux, Uy, pch = 20, col = "black")
ncap <- apply(y, 1, sum)
y <- y[ncap > 0, ]
Ux <- Ux[ncap > 0, ]
Uy <- Uy[ncap > 0, ]
M <- 200
nind <- nrow(y)
y <- rbind(y, matrix(0, nrow = (M - nrow(y)), ncol = ncol(y)))

# fill in "NA" values in the observed location data
x_fill_values <- matrix(NA, nrow = (M - nind), ncol = ncol(y))
y_fill_values <- matrix(NA, nrow = (M - nind), ncol = ncol(y))
Ux <- rbind(Ux, x_fill_values)
Uy <- rbind(Uy, y_fill_values)

# unroll the location data into long form to avoid NA values for Stan
i_mat <- matrix(rep(1:M, times = K), nrow = M)
k_mat <- matrix(rep(1:K, each = M), nrow = M)

n_detections <- sum(y)
n_unk <- sum(!y)

ux_obs <- Ux[y == 1]
uy_obs <- Uy[y == 1]
i_obs <- i_mat[y == 1]
k_obs <- k_mat[y == 1]

i_unk <- i_mat[y == 0]
k_unk <- k_mat[y == 0]

stopifnot(M*K == (n_detections + n_unk))

# Fit the model in Stan ---------------------------------------------------

stan_d <- list(y = y, 
               n_detections = n_detections,
               n_unk = n_unk,
               ux_obs = ux_obs,
               uy_obs = uy_obs,
               i_obs = i_obs,
               k_obs = k_obs,
               i_unk = i_unk,
               k_unk = k_unk,
               X = X, 
               n_occasion = K, 
               M = M, 
               n_point = J, 
               xlim = xlim, 
               ylim = ylim)

sx_means <- apply(Ux, 1, mean, na.rm = TRUE)
sy_means <- apply(Uy, 1, mean, na.rm = TRUE)

make_inits <- function(chain_id) {
  list(s1 = ifelse(is.na(sx_means), runif(M, xlim[1], xlim[2]), sx_means), 
       s2 = ifelse(is.na(sy_means), runif(M, ylim[1], ylim[2]), sy_means))
}

m_init <- stan_model("ch15/fixed-search-path.stan")
m_fit <- sampling(m_init, data = stan_d, init = make_inits)
# Lots of divergent transitions here - adapt_delta=0.99 doesn't fix it.
# Also, these results also differ from those presented in the book...dunno why.

traceplot(m_fit, pars = c("N", "alpha0", "alpha1", "psi", "sigma_move"))
print(m_fit, pars = c("N", "alpha0", "alpha1", "psi", "sigma_move"))


traceplot(m_fit, pars = c("s1"))
