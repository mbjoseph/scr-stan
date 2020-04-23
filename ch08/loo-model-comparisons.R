library(scrbook)
library(rstan)
library(loo)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)


# Comparing wolverine models with and without sex effects -----------------
# data processing code from scrbook::wolvSCR0ms()

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

area <- (Xu - Xl) * (Yu - Yl)/10
plot(traplocs, pch = 20, xlim = c(Xl, Xu), ylim = c(Yl, Yu))

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
               sex = wolverine$wsex + 1,
               n_trap = ntraps, 
               n_occasion = K, 
               X = traplocs, 
               y = y, 
               xlim = c(Xl, Xu), 
               ylim = c(Yl, Yu))

m_init <- stan_model("ch08/wolverine-sex-effects.stan")
m_fit <- sampling(m_init, data = stan_d)
rstan::traceplot(m_fit, pars = c("alpha0", "alpha1", "psi_sex", "N"))

m2_init <- stan_model("ch08/wolverine-no-sex-effects.stan")
m2_fit <- sampling(m2_init, data = stan_d)
rstan::traceplot(m2_fit, pars = c("alpha0", "alpha1", "N"))



# Compare models using approximate leave-one-out cross validation -------------

# Note Pareto k diagnostic values indicate this comparison may not be valid.
# I wonder what we could do to make this model more robust...
# maybe a beta-binomial observation model
(loo_sex <- loo(m_fit))
(loo_no_sex <- loo(m2_fit))
loo_compare(loo_sex, loo_no_sex)
