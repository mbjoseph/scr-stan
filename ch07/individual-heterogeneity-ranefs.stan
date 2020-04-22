
data {
  int<lower = 1> M;
  int<lower = 0> n_aug;
  int<lower = 0, upper = M> n_obs;
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion;
  matrix[n_trap, 2] X;
  int<lower = 0, upper = n_occasion> y[M, n_trap];
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
  real mu_alpha0;
  real<lower = 0> sd_alpha0;
  vector[M] z_alpha0;
  real<lower = 0> alpha1;
  real<lower = 0, upper = 1> psi;
  vector<lower = xlim[1], upper = xlim[2]>[M] s1;
  vector<lower = ylim[1], upper = ylim[2]>[M] s2;
}

transformed parameters {
  matrix[M, 2] s = append_col(s1, s2);
  vector[M] lp_if_present;
  
  {
    matrix[M, n_trap] dist_to_trap;
    matrix[M, n_trap] log_p;
    matrix[M, n_trap] logit_p;

    for (i in 1:M) {
      for (j in 1:n_trap) {
        dist_to_trap[i, j] = distance(s[i, ], X[j, ]);
        log_p[i, j] = log_inv_logit(mu_alpha0 + z_alpha0[i] * sd_alpha0) 
                         - alpha1 * dist_to_trap[i, j];
        logit_p[i, j] = log_p[i, j] - log1m_exp(log_p[i, j]);
      }
        lp_if_present[i] = bernoulli_lpmf(1 | psi)
          + binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[i, ]);
    }
  }
}

model {
  // priors
  mu_alpha0 ~ std_normal();
  sd_alpha0 ~ std_normal();
  z_alpha0 ~ std_normal();
  alpha1 ~ normal(0, 3);
  
  // likelihood
  for (i in 1:M) {
    if (observed[i]) {
      target += lp_if_present[i];
    } else {
      target += log_sum_exp(lp_if_present[i], bernoulli_lpmf(0 | psi));
    }
  }
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
