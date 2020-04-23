
data {
  int<lower = 1> M;
  int<lower = 0> n_aug;
  int<lower = 0, upper = M> n_obs;
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion[n_trap];
  matrix[n_trap, 2] X;
  int<lower = 0> y[M, n_trap];
  vector[2] xlim;
  vector[2] ylim;
}

transformed data {
  int<lower = 0, upper = 1> observed[M];

  for (i in 1:M) {
    if (sum(y[i, ]) > 0) {
      observed[i] = 1;
    } else {
      observed[i] = 0;
    }
  }
}

parameters {
  real alpha0;
  real<lower = 0> alpha1;
  real<lower = 0, upper = 1> psi;
  vector<lower = xlim[1], upper = xlim[2]>[M] s1;
  vector<lower = ylim[1], upper = ylim[2]>[M] s2;
}

transformed parameters {
  matrix[M, 2] s = append_col(s1, s2);
  vector[M] lp_if_present;
  vector[M] log_lik;
  
  {
    matrix[M, n_trap] sq_dist;
    matrix[M, n_trap] log_p; // last dim corresponds to sex
    matrix[M, n_trap] logit_p;

    for (i in 1:M) {
      for (j in 1:n_trap) {
        sq_dist[i, j] = squared_distance(s[i, ], X[j, ]);
        log_p[i, j] = log_inv_logit(alpha0) - alpha1 * sq_dist[i, j];
        logit_p[i, j] = log_p[i, j] - log1m_exp(log_p[i, j]);
      }
      lp_if_present[i] = bernoulli_lpmf(1 | psi)
        + binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[i, ]);
    }
    
    for (i in 1:M) {
      if (observed[i]) {
        log_lik[i] = lp_if_present[i];
      } else {
        log_lik[i] = log_sum_exp(lp_if_present[i], bernoulli_lpmf(0 | psi));
      }
    }
  } // end temp scope
}

model {
  // priors
  alpha0 ~ normal(0, 3);
  alpha1 ~ normal(0, 3);
  
  // likelihood
  target += sum(log_lik);
}


generated quantities {
  int N;
  
  {
    vector[M] lp_present; // [z=1][y=0 | z=1] / [y=0] on a log scale
    int z[M];

    for (i in 1:M) {
      if(observed[i]) {
        z[i] = 1;
      } else {
        lp_present[i] = lp_if_present[i]
                        - log_sum_exp(lp_if_present[i], 
                                      bernoulli_lpmf(0 | psi)
                                      );
        z[i] = bernoulli_rng(exp(lp_present[i]));
      }
    }
    N = sum(z);
  }
}
