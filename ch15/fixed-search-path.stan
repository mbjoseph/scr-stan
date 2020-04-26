
data {
  int<lower = 1> M;
  int<lower = 1> n_occasion;
  int<lower = 1> n_point;
  matrix[n_point, 2] X;
  int<lower = 0, upper = 1> y[M, n_occasion];
  vector[2] xlim;
  vector[2] ylim;
  
  // partly known individual location data
  int<lower = 0, upper = M * n_occasion> n_detections;
  vector<lower = xlim[1], upper = xlim[2]>[n_detections] ux_obs;
  vector<lower = ylim[1], upper = ylim[2]>[n_detections] uy_obs;
  int<lower = 1, upper = M> i_obs[n_detections];
  int<lower = 1, upper = n_occasion> k_obs[n_detections];
  int<lower = 0, upper = M * n_occasion> n_unk;
  int<lower = 1, upper = M> i_unk[n_unk];
  int<lower = 1, upper = n_occasion> k_unk[n_unk];
}

transformed data {
  int<lower = 0, upper = 1> observed[M];
  
  for (i in 1:M) {
    observed[i] = max(y[i, ]);
  }
}

parameters {
  real alpha0;
  real<lower = 0> alpha1;
  real<lower = 0, upper = 1> psi;
  real<lower = 0> sigma_move;
  
  vector<lower = xlim[1], upper = xlim[2]>[M] s1;
  vector<lower = ylim[1], upper = ylim[2]>[M] s2;
  vector<lower = xlim[1], upper = xlim[2]>[n_unk] ux_unk;
  vector<lower = ylim[1], upper = ylim[2]>[n_unk] uy_unk;
}

transformed parameters {
  matrix[M, 2] s = append_col(s1, s2);
  vector[M] lp_if_present;
  vector[M] log_lik;
  matrix[n_occasion, 2] u[M];
  
  for (i in 1:n_detections) {
    u[i_obs[i], k_obs[i], 1] = ux_obs[i];
    u[i_obs[i], k_obs[i], 2] = uy_obs[i];
  }
  
  for (i in 1:n_unk) {
    u[i_unk[i], k_unk[i], 1] = ux_unk[i];
    u[i_unk[i], k_unk[i], 2] = uy_unk[i];
  }
  
  {
    matrix[n_occasion, n_point] log_h;
    vector[n_occasion] log_H;
    vector[n_occasion] p;

    for (i in 1:M) {
      for (k in 1:n_occasion) {
        for (j in 1:n_point) {
          log_h[k, j] = - alpha1 * squared_distance(u[i, k], X[j, ]);
        }
        log_H[k] = alpha0 + log_sum_exp(log_h[k, ]); // eq. 15.2.2
        p[k] = inv_cloglog(log_H[k]);
      }
      
      lp_if_present[i] = bernoulli_lpmf(1 | psi) + bernoulli_lpmf(y[i, ] | p);
      
      if (observed[i]) {
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
  sigma_move ~ normal(0, 3);

  ux_obs ~ normal(s1[i_obs], sigma_move);
  uy_obs ~ normal(s2[i_obs], sigma_move);
  
  ux_unk ~ normal(s1[i_unk], sigma_move);
  uy_unk ~ normal(s2[i_unk], sigma_move);

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
