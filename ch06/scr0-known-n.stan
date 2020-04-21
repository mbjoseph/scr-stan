
data {
  int<lower = 1> n_individual;
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion;
  int<lower = 1> n_grid;
  matrix[n_grid, 2] grid_pts;
  matrix[n_trap, 2] X;
  int<lower = 0, upper = n_occasion> y[n_individual, n_trap];
  int<lower = 0> n_zero_histories;
}

transformed data {
  matrix[n_grid, n_trap] sq_dist;
  vector[n_individual] ever_observed;
  int n_nonzero_histories = n_individual - n_zero_histories;
  int y_zero[n_trap];
  real log_n_grid = log(n_grid);
  
  for (i in 1:n_grid) {
    for (j in 1:n_trap) {
      sq_dist[i, j] = squared_distance(grid_pts[i, ], X[j, ]);
    }
  }
  
  for (i in 1:n_individual) {
    if (sum(y[i, ]) > 0) {
      ever_observed[i] = 1;
    } else {
      ever_observed[i] = 0;
    }
  }
  
  for (i in 1:n_trap) {
    y_zero[i] = 0;
  }
}

parameters {
  real<lower = 0, upper = 1> alpha0;
  real alpha1;
}

transformed parameters {
  real log_p0 = log(alpha0);
  matrix[n_grid, n_trap] logit_p;
  vector[n_nonzero_histories] loglik_observed;
  real loglik_zero_history;
  
  {
    matrix[n_grid, n_trap] log_p = log_p0 - alpha1 * sq_dist;
    vector[n_grid] tmp;
    int counter=1;
    
    logit_p = log_p - log1m_exp(log_p);
    
    // nonzero encounter history probability
    for (i in 1:n_individual) {
      if (ever_observed[i]) {
        for (j in 1:n_grid) {
          tmp[j] = binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[j, ]);
        }
        loglik_observed[counter]= log_sum_exp(tmp) - log_n_grid;
        counter += 1;
      }
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
  alpha1 ~ normal(0, 3);
  
  // likelihood
  target += sum(loglik_observed);
  target += n_zero_histories * loglik_zero_history;
}
