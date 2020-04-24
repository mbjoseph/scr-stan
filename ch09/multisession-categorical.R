library(scrbook)
library(secr)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


# Data processing code from scrbook::SCRovenbird() ----------------
set.seed(2013)
data(ovenbird)

par(mfrow = c(1, 3))
plot(ovenCH[["2005"]])
plot(ovenCH[["2007"]])
plot(ovenCH[["2009"]])
par(mfrow = c(1, 1))

buffer <- 150
X <- traps <- traps(ovenCH)
xlim <- c(min(X[[1]][, 1]) - buffer, max(X[[1]][, 1]) + buffer)
ylim <- c(min(X[[1]][, 2]) - buffer, max(X[[1]][, 2]) + buffer)
ntraps <- nrow(traps[[1]])
Y <- ovenCH
K <- 10
M <- 100
Sst0 <- cbind(runif(M, xlim[1], xlim[2]), runif(M, ylim[1], 
                                                ylim[2]))
Sst <- NULL
Ycat <- NULL
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
  sout <- spiderplot(tmp2[1:nind, 1:nrep], as.matrix(X[[i]]))$avg.s
  Stmp <- Sst0
  Stmp[1:nind, 1:2] <- sout
  Sst <- rbind(Sst, Stmp)
  if (nrow(xx) < 10) {
    # for years with < 10 sampling occasions, fill Y with zeros (to act as NAs)
    # Original scrbook pkg uses NA values, but these aren't allowed in Stan.
    # However, for the categorical distribution in Stan, y=0 will raise an 
    # error -- therefore 0 is a good NA value to fill this array with. 
    # If (by accident) these NA observations are passed to the likelihood, we 
    # would get an error in Stan.
    tmp2[1:M, (nrow(xx) + 1):10] <- 0
  }
  Ycat <- rbind(Ycat, tmp2)
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

stan_d <- list(
  M = M, 
  n_year = n_year,
  bigM = nrow(Ycat),
  year = rep(1:n_year, each = M),
  n_trap = nrow(X[[1]]), 
  n_occasion = unlist(lapply(Y, function(x) dim(x)[2])), 
  max_n_occasion = K, # first year only has 9 occasions, rest have 10
  X = as.matrix(X[[1]]), 
  y = Ycat, 
  known_dead = died,
  xlim = xlim, 
  ylim = ylim
)

m_init <- stan_model("ch09/multisession-categorical.stan")
m_fit <- sampling(m_init, data = stan_d, chains = 4)

# These results are slightly different from the ones in the book (table 9.2), 
# but the results in that table seem weird. If M=100 (in each year), 
# then how is it possible that the 97.5% posterior quantiles for N are > 100?
traceplot(m_fit, pars = c("alpha0", "alpha1", "psi", "N"))
pairs(m_fit, pars = c("alpha0", "alpha1", "psi"))
print(m_fit, pars = c("alpha0", "alpha1", "psi", "N"))
