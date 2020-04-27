
data {
  int<lower = 1> M;
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion;
  int<lower = 1> n_grid;
  matrix[n_grid, 2] grid_pts;
  matrix[n_trap, 2] X;
  int<lower = 0, upper = n_occasion> y[M, n_trap];
  int<lower = 0> n_nonzero_histories;
  int<lower = 0> n_zero_histories;
}

transformed data {
  matrix[n_grid, n_trap] sq_dist;
  vector[M] ever_observed;
  real log_n_grid = log(n_grid);

  for (i in 1:n_grid) {
    for (j in 1:n_trap) {
      sq_dist[i, j] = squared_distance(grid_pts[i, ], X[j, ]);
    }
  }
  
  for (i in 1:M) {
    if (sum(y[i, ]) > 0) {
      ever_observed[i] = 1;
    } else {
      ever_observed[i] = 0;
    }
  }
}

parameters {
  real alpha0;
  real alpha1;
  real<lower = 0, upper = 1> psi;
}

transformed parameters {
  real log_psi = log(psi);
  real log1m_psi = log1m(psi);
  matrix[n_grid, n_trap] log_p = log_inv_logit(alpha0) - alpha1 * sq_dist;
  matrix[n_grid, n_trap] logit_p = log_p - log1m_exp(log_p);
  vector[M] lp_if_present;
  vector[M] log_lik;

  {
    vector[n_grid] tmp;
    
    for (i in 1:M) {
      for (j in 1:n_grid) {
        tmp[j] = binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[j, ]);
      }
      lp_if_present[i] = log_psi + log_sum_exp(tmp) - log_n_grid;
      if (ever_observed[i]) {
        log_lik[i] = lp_if_present[i];
      } else {
        log_lik[i] = log_sum_exp(lp_if_present[i], log1m_psi);
      }
    }
  }
}

model {
  // priors
  alpha0 ~ normal(0, 3);
  alpha1 ~ normal(0, 3);
  
  // likelihood
  target += sum(log_lik);
}


generated quantities {
  int<lower = n_nonzero_histories, upper = M> N;
  
  {
    matrix[n_grid, n_trap] p = inv_logit(logit_p);
    vector[n_zero_histories] lp_present;
    int z_aug[n_zero_histories];
    vector[n_grid] tmp;

    for (i in 1:n_zero_histories) {
      for (j in 1:n_grid) {
        tmp[j] = binomial_logit_lpmf(y[n_nonzero_histories + i, ] | n_occasion, logit_p[j, ]);
      }

      // lp_present is [z=1][y=0 | z=1] / [y=0] on a log scale
      lp_present[i] = log_psi + log_sum_exp(tmp) - log_n_grid
                      - log_sum_exp(log_psi + log_sum_exp(tmp) - log_n_grid, 
                                    log1m_psi);
      z_aug[i] = bernoulli_rng(exp(lp_present[i]));
    }
    N = n_nonzero_histories + sum(z_aug);
  }
}
