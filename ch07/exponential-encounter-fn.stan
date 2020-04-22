
data {
  int<lower = 1> n_nonzero_histories;
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion;
  int<lower = 1> n_grid;
  matrix[n_grid, 2] grid_pts;
  matrix[n_trap, 2] X;
  int<lower = 0, upper = n_occasion> y[n_nonzero_histories, n_trap];
  real n0scale;
}

transformed data {
  matrix[n_grid, n_trap] dist;
  int y_zero[n_trap];
  real log_n_grid = log(n_grid);
  
  for (i in 1:n_grid) {
    for (j in 1:n_trap) {
      dist[i, j] = distance(grid_pts[i, ], X[j, ]);
    }
  }

  for (i in 1:n_trap) {
    y_zero[i] = 0;
  }
}

parameters {
  real alpha0;
  real alpha1;
  real<lower = 0> n0;
}

transformed parameters {
  real<lower = n_nonzero_histories> N = n_nonzero_histories + n0;
  matrix[n_grid, n_trap] log_p = log_inv_logit(alpha0) - alpha1 * dist;
  matrix[n_grid, n_trap] logit_p = log_p - log1m_exp(log_p);
  vector[n_nonzero_histories] loglik_observed;
  real loglik_zero_history;
  
  {
    vector[n_grid] tmp;

    // nonzero encounter history probability
    for (i in 1:n_nonzero_histories) {
        for (j in 1:n_grid) {
          tmp[j] = binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[j, ]);
        }
        loglik_observed[i]= log_sum_exp(tmp) - log_n_grid;
    }
    
    // all zero encounter history probability
    for (j in 1:n_grid) {
      tmp[j] = binomial_logit_lpmf(y_zero | n_occasion, logit_p[j, ]);
    }
    loglik_zero_history = log_sum_exp(tmp) - log_n_grid;
  }
}

model {
  // priors
  alpha0 ~ normal(0, 3);
  alpha1 ~ normal(0, 3);
  n0 ~ normal(0, n0scale);

  // likelihood
  target += lgamma(N + 1) - lgamma(n0 + 1);
  target += sum(loglik_observed);
  target += n0 * loglik_zero_history;
}


generated quantities {
  matrix[n_nonzero_histories, 2] s;
  
  {
    vector[n_grid] tmp;
    vector[n_grid] theta;
    for (i in 1:n_nonzero_histories) {
      for (j in 1:n_grid) {
        tmp[j] = binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[j, ]);
      }
    theta = exp(tmp - log_sum_exp(tmp));
    for (k in 1:2)
      s[i, k] = grid_pts[categorical_rng(theta), k];
    }
  }
}
