
data {
  int<lower = 1> n_individual;
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion;
  vector[2] xlim;
  vector[2] ylim;
  matrix[n_trap, 2] X;
  int<lower = 0, upper = n_occasion> y[n_individual, n_trap];
}

parameters {
  real alpha0;
  real alpha1;
  // bounds on s imply uniform priors over support
  vector<lower = xlim[1], upper = xlim[2]>[n_individual] s1;
  vector<lower = ylim[1], upper = ylim[2]>[n_individual] s2;
}

transformed parameters {
  matrix[n_individual, 2] s = append_col(s1, s2);
  matrix[n_individual, n_trap] logit_p;
  vector[n_individual] loglik;
  matrix[n_individual, n_trap] sq_dist;
  
  for (i in 1:n_individual) {
    for (j in 1:n_trap) {
      sq_dist[i, j] = squared_distance(s[i, ], X[j, ]);
    }
  }

  {
    matrix[n_individual, n_trap] log_p = log_inv_logit(alpha0) - alpha1 * sq_dist;
    logit_p = log_p - log1m_exp(log_p);

    // nonzero encounter history probability
    for (i in 1:n_individual) {
      loglik[i]= binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[i, ]);
    }
  }
}

model {
  // priors
  alpha0 ~ normal(0, 3);
  alpha1 ~ normal(0, 3);
  
  target += sum(loglik);
}
