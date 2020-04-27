
data {
  int<lower = 1> M;
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion;
  matrix[n_trap, 2] X;
  int<lower = 1, upper = n_trap + 1> y[M, n_occasion];
  vector[2] xlim;
  vector[2] ylim;
}

transformed data {
  int<lower = 0, upper = 1> observed[M];

  for (i in 1:M) {
    if (min(y[i, ]) < (n_trap + 1)) {
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
    matrix[M, n_trap] dist;
    matrix[M, n_trap + 1] logits; // last element is "not detected"

    for (i in 1:M) {
      for (j in 1:n_trap) {
        dist[i, j] = distance(s[i, ], X[j, ]);
        logits[i, j] = alpha0 - alpha1 * dist[i, j];
      }
      logits[i, n_trap + 1] = 0;
      lp_if_present[i] = bernoulli_lpmf(1 | psi)
        + categorical_logit_lpmf(y[i, ] | to_vector(logits[i, ]));
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
        lp_present[i] = lp_if_present[i] - log_lik[i];
        z[i] = bernoulli_rng(exp(lp_present[i]));
      }
    }
    N = sum(z);
  }
}
