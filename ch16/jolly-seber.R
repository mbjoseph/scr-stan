library(secr)
library(scrbook)
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
# adapted from scrbook::ovenbirds.js

data(ovenbird)
X <- traps <- traps(ovenCH)
xlim <- c(min(X[[1]][, 1]) - 150, max(X[[1]][, 1]) + 150)
ylim <- c(min(X[[1]][, 2]) - 150, max(X[[1]][, 2]) + 150)
ntraps <- nrow(traps[[1]])
Y <- ovenCH
K <- 10
M <- 200
Sst <- cbind(runif(M, xlim[1], xlim[2]), runif(M, ylim[1],
                                               ylim[2]))
Sst <- array(Sst, dim = c(M, 2, 5))
hold <- unique(c(unlist(dimnames(Y[[1]])[1]), unlist(dimnames(Y[[2]])[1]), 
                 unlist(dimnames(Y[[3]])[1]), unlist(dimnames(Y[[4]])[1]), 
                 unlist(dimnames(Y[[5]])[1])))
Yarr <- array(ntraps + 1, dim = c(M, K, 5))
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
  tmp[tmp < 0] <- tmp[tmp < 0] * (-1)
  tmp[tmp == 0] <- ntraps + 1
  nind <- nrow(tmp)
  nrep <- ncol(tmp)
  tmp2 <- matrix(ntraps + 1, nrow = M, ncol = 10)
  tmp2[pmatch(unlist(dimnames(Y[[i]])[1]), hold), 1:nrep] <- tmp
  Stmp <- Sst[, , i]
  sbar <- spiderplot(tmp2[pmatch(unlist(dimnames(Y[[i]])[1]), 
                                 hold), 1:nrep], as.matrix(X[[i]]))$avg.s
  Stmp[pmatch(unlist(dimnames(Y[[i]])[1]), hold), 1:2] <- sbar
  Sst[, , i] <- Stmp
  Yarr[, , i] <- tmp2
  if (nrow(xx) < 10) {
    Yarr[1:M, (nrow(xx) + 1):10, i] <- NA
  }
}

# Convert detection data to categorical: 0: NA, 1: detected, 2: not detected
y_binary_cat <- Yarr
y_binary_cat[is.na(y_binary_cat)] <- 0
y_binary_cat[y_binary_cat < 45] <- 1
y_binary_cat[y_binary_cat == 45] <- 2
table(y_binary_cat)
stan_d <- list(y = y_binary_cat, M = M, T = 5, K = c(9, 10, 10, 10, 10), Kmax = 10)


# Fit the nonspatial model ---------------------------------------------------

m_init <- stan_model("ch16/nonspatial-jolly-seber.stan")
m_fit <- sampling(m_init, data = stan_d) 

# we get R-hat warnings, but these could be accounted for by fixed 
# transformed parameters. best to check manually:
traceplot(m_fit, pars = c("p", "phi", "psi", "gamma"))
print(m_fit, pars = c("p", "phi", "psi", "gamma"))



# Fit the spatial JS model ------------------------------------------------

sjs_d <- stan_d
sjs_d$y <- Yarr
sjs_d$y[is.na(sjs_d$y)] <- 0 # using 0 as NA
sjs_d$X <- as.matrix(X[[1]])
sjs_d$n_trap <- ntraps
sjs_d$xlim <- xlim
sjs_d$ylim <- ylim

s_init <- stan_model("ch16/spatial-jolly-seber.stan")
# good inits are important for this model -- alpha1 needs to be small
s_fit <- sampling(s_init, data = sjs_d, init_r = 1,
                  init = function() list(alpha1 = .001, p0 = .1), 
                  control = list(max_treedepth = 11))
# We will get Rhat warnings because of partly observed params. 
# Default params result in bulk and tail effective sample size warnings - 
# running more iterations may help.
traceplot(s_fit, pars = c("gamma", "psi", "phi", "p0", "alpha1"))
pairs(s_fit, pars = c("gamma", "psi", "phi", "p0", "alpha1"))
print(s_fit, pars = c("gamma", "psi", "phi", "p0", "alpha1"))
