
data {
  int<lower = 1> M;
  int<lower = 1> n_trap;
  int<lower = 1> n_grid;
  matrix[n_grid, 2] grid_pts;
  vector[n_grid] canopy;
  matrix[n_trap, 2] X;
  int<lower = 0> y[M, n_trap];
  real pixel_area;
}

transformed data {
  real log_pixel_area = log(pixel_area);
  matrix[n_grid, n_trap] sq_dist;
  vector[M] ever_observed;
  real logM = log(M);

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
  real beta0;
  real beta1;
  real loglam0;
  real<lower = 0> alpha1;
}

transformed parameters {
  real log_en;
  real<upper = 0> log_psi;
  real<upper = 0> log1m_psi;
  vector[M] lp_if_present;
  vector[M] loglik;

  {
    vector[n_grid] p_g;
    vector[n_grid] log_mu = beta0 + beta1 * canopy + log_pixel_area;
    vector[n_grid] log_probs = log_softmax(log_mu);
    
    log_en = log_sum_exp(log_mu);
    log_psi = log_en - logM;
    log1m_psi = log1m_exp(log_psi);
    
    for (i in 1:M) {
      for (j in 1:n_grid) {
        // p_g[i, j] = log p(y[i, ]  | s[i] = j, z = 1, ...) p(s[i] = j)
        p_g[j] = poisson_log_lpmf(y[i, ] | loglam0 - alpha1 * sq_dist[j, ])
                 + log_probs[j];
      }
      lp_if_present[i] = log_psi + log_sum_exp(p_g);
      if (ever_observed[i]) {
        loglik[i] = lp_if_present[i];
      } else {
        loglik[i] = log_sum_exp(lp_if_present[i], log1m_psi);
      }
    }
  }
}

model {
  // priors
  beta0 ~ normal(0, 3);
  beta1 ~ normal(0, 3);
  loglam0 ~ normal(0, 3);
  alpha1 ~ normal(0, 3);
  
  // likelihood
  target += sum(loglik);
}


generated quantities {
  int<upper = M> N;
  
  {
    vector[M] lp_present; // [z=1][y=0 | z=1] / [y=0] on a log scale
    int z[M];

    for (i in 1:M) {
      if(ever_observed[i]) {
        z[i] = 1;
      } else {
        lp_present[i] = lp_if_present[i]
                        - log_sum_exp(lp_if_present[i], log1m_psi);
        z[i] = bernoulli_rng(exp(lp_present[i]));
      }
    }
    N = sum(z);
  }
}
