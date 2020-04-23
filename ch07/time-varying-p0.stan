
data {
  int<lower = 1> M;
  int<lower = 0> n_aug;
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion;
  matrix[n_trap, 2] X;
  int<lower = 0, upper = 1> y[M, n_trap, n_occasion];
  vector[2] xlim;
  vector[2] ylim;
}

transformed data {
  int<lower = 0, upper = M> n_obs = M - n_aug;
  int<lower = 0, upper = 1> detected[M];
  
  for (i in 1:M) {
    detected[i] = 0;
    for (j in 1:n_trap) {
      for (k in 1:n_occasion) {
        if (y[i, j, k] > 0) {
          detected[i] = 1;
        }
      }
    }
  }
}

parameters {
  real mu_alpha0;
  real<lower = 0> sd_alpha0;
  vector[n_occasion] z_alpha0;
  real<lower = 0> alpha1;
  real<lower = 0, upper = 1> psi;
  vector<lower = xlim[1], upper = xlim[2]>[M] s1;
  vector<lower = ylim[1], upper = ylim[2]>[M] s2;
}

transformed parameters {
  matrix[M, 2] s = append_col(s1, s2);
  vector[n_occasion] alpha0;
  vector[n_occasion] p0;
  vector[M] lp_if_present;
  
  alpha0 = mu_alpha0 + z_alpha0 * sd_alpha0;
  p0 = inv_logit(alpha0);
  
  { // begin temporary scope
    matrix[n_trap, n_occasion] logit_p[M];
    matrix[M, n_trap] sq_dist;
    real log_p;

    for (i in 1:M) {
      for (j in 1:n_trap) {
        sq_dist[i, j] = squared_distance(s[i, ], X[j, ]);
        for (k in 1:n_occasion) {
          log_p = log_inv_logit(alpha0[k]) - alpha1 * sq_dist[i, j];
          logit_p[i, j, k] = log_p - log1m_exp(log_p);
        }
      }
    }
    
    for (i in 1:M) {
      lp_if_present[i] = bernoulli_lpmf(1 | psi);
      for (k in 1:n_occasion) {
        lp_if_present[i] += bernoulli_logit_lpmf(y[i, , k] | logit_p[i, , k]);
      }
    }
  } // end temporary scope
}

model {
  // priors
  mu_alpha0 ~ std_normal();
  sd_alpha0 ~ std_normal();
  z_alpha0 ~ std_normal();
  alpha1 ~ normal(0, 3);
  
  // likelihood
  for (i in 1:M) {
    if (detected[i]) {
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
      if (detected[i]) {
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
