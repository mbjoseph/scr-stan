library(scrbook)
library(rstan)
library(secr)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# adapted from scrbook::ovenbird.ms
data(ovenbird)
set.seed(2013)

X <- traps <- traps(ovenCH)
xlim <- c(min(X[[1]][, 1]) - 300, max(X[[1]][, 1]) + 300)
ylim <- c(min(X[[1]][, 2]) - 300, max(X[[1]][, 2]) + 300)
ntraps <- nrow(traps[[1]])
Y <- ovenCH
K <- 10
M <- 200
Sst0 <- cbind(runif(M, xlim[1], xlim[2]), runif(M, ylim[1], 
                                                ylim[2]))
Sst <- NULL
Ymat <- NULL
died <- NULL
for (i in 1:5) {
  tmp <- Y[[i]]
  Y2d <- matrix(0, nrow = dim(tmp)[1], ncol = K)
  for (ind in 1:dim(tmp)[1]) {
    xx <- tmp[ind, , ]
    for (j in 1:nrow(xx)) {
      if (any(xx[j, ] > 0)) 
        Y2d[ind, j] <- (1:ncol(xx))[xx[j, ] > 0]
      if (any(xx[j, ] < 0)) 
        Y2d[ind, j] <- (1:ncol(xx))[xx[j, ] < 0] * 
          (-1)
    }
  }
  tmp <- Y2d
  nind <- nrow(tmp)
  nrep <- ncol(tmp)
  D <- matrix(0, nrow = M, ncol = 10)
  D[1:nind, 1:nrep] <- tmp
  D[D > 0] <- 0
  D[D < 0] <- 1
  died <- rbind(died, D)
  tmp[tmp < 0] <- 0
  tmp[tmp == 0] <- ntraps + 1
  tmp2 <- matrix(NA, nrow = M, ncol = 10)
  tmp2[, 1:nrep] <- ntraps + 1
  tmp2[1:nind, 1:nrep] <- tmp
  if (nrow(xx) < 10) {
    # for years with < 10 sampling occasions, fill Y with zeros (to act as NAs)
    # Original scrbook pkg uses NA values, but these aren't allowed in Stan.
    # However, for the categorical distribution in Stan, y=0 will raise an 
    # error -- therefore 0 is a good NA value to fill this array with. 
    # If (by accident) these NA observations are passed to the likelihood, we 
    # would get an error in Stan.
    tmp2[1:M, (nrow(xx) + 1):10] <- 0
  }
  Ymat <- rbind(Ymat, tmp2)
  sout <- spiderplot(tmp2[1:nind, 1:nrep], as.matrix(X[[i]]))$avg.s
  Stmp <- Sst0
  Stmp[1:nind, 1:2] <- sout
  Sst <- rbind(Sst, Stmp)
}


for (i in 1:nrow(died)) {
  xx <- died[i, ]
  if (sum(xx) > 0) {
    first <- (1:length(xx))[xx == 1]
    died[i, first:ncol(died)] <- 1
    died[i, first] <- 0
  }
}

n_year <- 5

nind <- c(nrow(Y[[1]]), nrow(Y[[2]]), nrow(Y[[3]]), nrow(Y[[4]]), 
          nrow(Y[[5]]))
yrid <- NULL
for (i in 1:5) {
  yrid <- c(yrid, rep(i, nind[i]), rep(0, M - nind[i]))
}

# Replace 0 (NA) values with n_trap + 1 for individuals never observed.
# Because we don't know what year they might have occurred in, we do not
# know how many sampling occasions there were. 
n_trap <- nrow(X[[1]])
for (i in seq_along(yrid)) {
  if (yrid[i] == 0) {
    Ymat[i, ] <- n_trap + 1
  }
}

stan_d <- list(
  M = M, 
  n_year = n_year,
  bigM = nrow(Ymat),
  year_id = yrid,
  n_trap = n_trap, 
  n_occasion = unlist(lapply(Y, function(x) dim(x)[2])), 
  max_n_occasion = K, # first year only has 9 occasions, rest have 10
  X = as.matrix(X[[1]]), 
  y = Ymat, 
  known_dead = died,
  xlim = xlim, 
  ylim = ylim
)

m_init <- stan_model("ch14/ovenbird-multisession.stan")
m_fit <- sampling(m_init, data = stan_d)
traceplot(m_fit, pars = c("alpha0", "alpha1", "N"))
print(m_fit, pars = c("alpha0", "alpha1", "N", "beta0", "psi"))
# Note that these values are somewhat different than those in the book.
