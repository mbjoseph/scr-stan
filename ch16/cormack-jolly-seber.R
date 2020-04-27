library(secr)
library(scrbook)
library(rstan)
library(reshape)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
# adapted from scrbook::shad.cjs

data(Ch16shaddata)
shad <- Ch16shaddata$shad
melted.rkm <- melt(shad, id = c("TagID", "RKM"))
tagid.week <- cast(melted.rkm, TagID ~ RKM ~ value, fill = 0, 
                   length)
hold = as.matrix(table(shad$TagID, shad$Week))
hold[hold > 1] <- 1
y <- hold
melt.state <- melt(shad, id = c("TagID", "state", "Week"))
shad.state <- cast(melt.state, TagID ~ Week ~ state, fill = 0, 
                   length)
newy = matrix(NA, 315, 12)
for (i in 1:315) {
  for (j in 1:12) {
    a = which(shad.state[i, j, ] == max(shad.state[i, 
                                                   j, ]))
    newy[i, j] <- as.numeric(a[1])
    if (max(shad.state[i, j, ]) == 0) 
      newy[i, j] <- 3
  }
}
first <- Ch16shaddata$first
M <- 315
T <- 12
nantenna <- 7
antenna.loc <- c(3.7, 7.7, 13.4, 45.3, 56.4, 72, 77)
xl <- 0
xu <- 82

ycat <- ifelse(y == 1, 1, 2)

stan_d <- list(
  M = nrow(ycat), 
  T = ncol(ycat), 
  y = ycat, 
  first_capture = first
)



# Fit the non-spatial model -----------------------------------------------

m_init <- stan_model("ch16/nonspatial-cormack-jolly-seber.stan")
m_fit <- sampling(m_init, data = stan_d)
traceplot(m_fit, pars = c("p", "phi"))


# Fit the spatial model ---------------------------------------------------
scjs_d <- list(
  y = tagid.week, 
  ymax = max(tagid.week),
  first_capture = first, 
  M = M, 
  T = T, 
  xlo = xl, 
  xhi = xu,
  ntrap = nantenna,
  trap_x = antenna.loc)
s_init <- stan_model("ch16/spatial-cormack-jolly-seber.stan")
s_fit <- sampling(s_init, data = scjs_d, init_r = .1,
                  init = function() {
                    list(alpha1 = runif(1, .001, .01), 
                         p0 = runif(1, .05, .3))
                  }
                  )
traceplot(s_fit, pars = c("p0", "alpha1", "phi"))
pairs(s_fit, pars = c("p0", "alpha1", "phi"))
print(s_fit, pars = c("p0", "alpha1", "phi"))
