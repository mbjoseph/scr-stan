
data {
  int<lower = 1> M;
  int<lower = 0> n_group;
  int<lower = 0, upper = n_group> obs_g[M]; // 0 acts as NA
  int<lower = 1> n_occasion;
  vector[n_group] x;
  int<lower = 0, upper = n_occasion> y[M];
}

transformed data {
  int<lower = 0, upper = 1> observed[M];
  real logM = log(M);
  
  for (i in 1:M) {
    observed[i] = y[i] > 0;
  }
}

parameters {
  real beta0;
  real beta1;
  real<lower = 0, upper = 1> p_detect;
}

transformed parameters {
  vector[n_group] log_lambda = beta0 + beta1 * x;
  real<upper = 0> log_psi = log_sum_exp(log_lambda) - logM;
  real<upper = 0> log1m_psi = log1m_exp(log_psi);
  vector[M] lp_if_present;
  vector[M] loglik;
  
  for (i in 1:M) {
    if (observed[i]) {
      lp_if_present[i] = log_psi 
                  + categorical_logit_lpmf(obs_g[i] | log_lambda)
                  + binomial_lpmf(y[i] | n_occasion, p_detect);
      loglik[i] = lp_if_present[i];
    } else {
      // we don't know the group, so we marginalize, which means computing
      // [y | z=1, g=1, p] [g=1] + ... + [y | z=1, g=G, p] [g=G].
      // But since [y | z=1, g, p] is equal for all g, this factors:
      // = [y | z=1, p] ([g=1] + ... + [g=G]). The terms in () sum to 1, so we 
      // are left with [y | z=1, p].
      lp_if_present[i] = log_psi + binomial_lpmf(y[i] | n_occasion, p_detect);
      loglik[i] = log_sum_exp(lp_if_present[i], log1m_psi);
    }
  }
}

model {
  // priors
  beta0 ~ normal(0, 3);
  beta1 ~ normal(0, 3);
  
  // likelihood
  target += sum(loglik);
}

generated quantities {
  int<upper = M> N;
  int<lower = 0, upper = M> Ng[n_group];
  int<lower = 1, upper = n_group> g[M];
  
  {
    vector[M] lp_present;
    int z[M];

    for (i in 1:M) {
      if(observed[i]) {
        z[i] = 1;
        g[i] = obs_g[i];
      } else {
        // [z=1][y=0 | z=1] / [y=0] on a log scale
        lp_present[i] = lp_if_present[i] - loglik[i];
        z[i] = bernoulli_rng(exp(lp_present[i]));
        g[i] = categorical_logit_rng(log_lambda);
      }
    }
    N = sum(z);
    
    for (group in 1:n_group) {
      Ng[group] = 0;
      for (i in 1:M) {
        if (z[i]) {
          if (g[i] == group) {
            Ng[group] += 1;
          }
        }
      }
    }
  }
}
