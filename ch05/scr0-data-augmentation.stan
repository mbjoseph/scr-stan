
data {
  int<lower = 1> M;
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion;
  matrix[n_trap, 2] X;
  int<lower = 0, upper = n_occasion> y[M, n_trap];
  vector[2] xlim;
  vector[2] ylim;
}

parameters {
  real alpha0;
  real<lower = 0> alpha1;
  real<lower = 0, upper = 1> psi;
  // bounds on s imply uniform priors over support
  vector<lower = xlim[1], upper = xlim[2]>[M] s1;
  vector<lower = ylim[1], upper = ylim[2]>[M] s2;
}

transformed parameters {
  matrix[M, n_trap] logit_p;
  matrix[M, 2] s = append_col(s1, s2);
  vector[M] lp_if_present;
  vector[M] log_lik;
  
  {
    matrix[M, n_trap] sq_dist;
    matrix[M, n_trap] log_p;

    for (i in 1:M) {
      for (j in 1:n_trap) {
        sq_dist[i, j] = squared_distance(s[i, ], X[j, ]);
      }
    }
    
    log_p = log_inv_logit(alpha0) - alpha1 * sq_dist;
    logit_p = log_p - log1m_exp(log_p);
    
    for (i in 1:M) {
      lp_if_present[i] = bernoulli_lpmf(1 | psi)
                       + binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[i, ]);
      if (sum(y[i, ]) > 0) {
        log_lik[i] = lp_if_present[i];
      } else {
        log_lik[i] = log_sum_exp(lp_if_present[i], bernoulli_lpmf(0 | psi));
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
  int<upper = M> N;
  
  {
    real lp_present;
    int z[M];
    
    N = 0;
    for (i in 1:M) {
      if (sum(y[i, ]) > 0) {
        z[i] = 1;
      } else {
        // lp_present is [z=1][y=0 | z=1] / [y=0] on a log scale
        lp_present = lp_if_present[i] - log_lik[i];
        z[i] = bernoulli_rng(exp(lp_present));
      }
      N += z[i];
    }
  }
}
