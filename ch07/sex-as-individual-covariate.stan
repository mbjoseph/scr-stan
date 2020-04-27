
data {
  int<lower = 1> M;
  int<lower = 0> n_aug;
  int<lower = 0, upper = M> n_obs;
  int<lower = 1, upper = 2> sex[n_obs]; // 1: male, 2: female
  int<lower = 1> n_trap;
  int<lower = 1> n_occasion;
  matrix[n_trap, 2] X;
  int<lower = 0, upper = n_occasion> y[M, n_trap];
  vector[2] xlim;
  vector[2] ylim;
}

transformed data {
  int<lower = 0, upper = 1> observed[M];
  int<lower = 0, upper = 1> is_female[n_obs];
  
  for (i in 1:M) {
    if (sum(y[i, ]) > 0) {
      observed[i] = 1;
    } else {
      observed[i] = 0;
    }
  }
  
  for (i in 1:n_obs) {
    is_female[i] = sex[i] - 1;
  }
}

parameters {
  vector[2] alpha0;
  real<lower = 0> alpha1[2];
  real<lower = 0, upper = 1> psi;
  real<lower = 0, upper = 1> psi_sex; // pr(sex = female)
  vector<lower = xlim[1], upper = xlim[2]>[M] s1;
  vector<lower = ylim[1], upper = ylim[2]>[M] s2;
}

transformed parameters {
  matrix[M, 2] s = append_col(s1, s2);
  vector[M] lp_if_present;
  vector[M] log_lik;
  
  {
    matrix[M, n_trap] sq_dist;
    matrix[n_trap, 2] log_p[M]; // last dim corresponds to sex
    matrix[n_trap, 2] logit_p[M];

    for (i in 1:M) {
      for (j in 1:n_trap) {
        sq_dist[i, j] = squared_distance(s[i, ], X[j, ]);
        for (k in 1:2) {
          log_p[i, j, k] = log_inv_logit(alpha0[k]) 
                           - alpha1[k] * sq_dist[i, j];
          logit_p[i, j, k] = log_p[i, j, k] - log1m_exp(log_p[i, j, k]);
        }
      }
      if (observed[i]) {
        // observed individual of known sex
        lp_if_present[i] = bernoulli_lpmf(1 | psi)
          + bernoulli_lpmf(is_female | psi_sex)
          + binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[i, , sex[i]]);
        log_lik[i] = lp_if_present[i];
      } else {
        // augmented individual of unknown sex
        lp_if_present[i] = bernoulli_lpmf(1 | psi)
          + log_mix(
              psi_sex, // pr(female)
              binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[i, , 2]), // ♀
              binomial_logit_lpmf(y[i, ] | n_occasion, logit_p[i, , 1])  // ♂
            );
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
