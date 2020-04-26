functions {
  real lstep(real x, real steepness, real minx, real maxx) {
    // log-scale continuous approximation to I(minx < x < maxx)
    real f1 = -log1p_exp(-steepness * (x - minx));
    real f2 = -log1p_exp(-steepness * (x - maxx));
    return f1 + log1m_exp(f2);
  }
}

data {
  int<lower = 1> M;
  int<lower = 1> n_occasion;
  int<lower = 0, upper = 1> y[M, n_occasion];
  vector[2] xlim;
  vector[2] ylim;
  
  // observed individual location data
  int<lower = 0, upper = M * n_occasion> n_detections;
  vector<lower = xlim[1], upper = xlim[2]>[n_detections] ux_obs;
  vector<lower = ylim[1], upper = ylim[2]>[n_detections] uy_obs;
  int<lower = 1, upper = M> i_obs[n_detections];
  int<lower = 1, upper = n_occasion> k_obs[n_detections];
  int<lower = 0, upper = M * n_occasion> n_unk;
  int<lower = 1, upper = M> i_unk[n_unk];
  int<lower = 1, upper = n_occasion> k_unk[n_unk];
  
  // parameters for the continuous approximation of a step function
  real phi;
  vector<lower = xlim[1], upper = xlim[2]>[2] search_x;
  vector<lower = ylim[1], upper = ylim[2]>[2] search_y;
}

transformed data {
  int<lower = 0, upper = 1> observed[M];
  int Kminus1 = n_occasion - 1;
  
  for (i in 1:M) {
    observed[i] = max(y[i, ]);
  }
}

parameters {
  real<lower = 0, upper = 1> p0;
  real<lower = 0, upper = 1> psi;
  real<lower = 0> sigma_move;
  vector<lower = xlim[1], upper = xlim[2]>[n_unk] ux_unk;
  vector<lower = ylim[1], upper = ylim[2]>[n_unk] uy_unk;
}

transformed parameters {
  vector[M] lp_if_present;
  vector[M] log_lik;
  matrix[n_occasion, 2] u[M];
  real log_p0 = log(p0);
  
  for (i in 1:n_detections) {
    u[i_obs[i], k_obs[i], 1] = ux_obs[i];
    u[i_obs[i], k_obs[i], 2] = uy_obs[i];
  }
  
  for (i in 1:n_unk) {
    u[i_unk[i], k_unk[i], 1] = ux_unk[i];
    u[i_unk[i], k_unk[i], 2] = uy_unk[i];
  }
  
  {
    vector[n_occasion] log_inside;
    vector[n_occasion] log_p;
    vector[n_occasion] logit_p;

    for (i in 1:M) {
      
      for (k in 1:n_occasion) {
        log_inside[k] = lstep(u[i, k, 1], phi, search_x[1], search_x[2])
                        + lstep(u[i, k, 2], phi, search_y[1], search_y[2]);
        log_p[k] = log_p0 + log_inside[k];
        logit_p[k] = log_p[k] - log1m_exp(log_p[k]);
      }
      
      lp_if_present[i] = bernoulli_lpmf(1 | psi) 
                         + bernoulli_logit_lpmf(y[i, ] | logit_p);
      
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
  log_p0 ~ normal(0, 3);
  sigma_move ~ std_normal();

  for (i in 1:M) {
    // initial location
    u[i, 1, 1] ~ uniform(xlim[1], xlim[2]);
    u[i, 1, 2] ~ uniform(ylim[1], ylim[2]);
    
    // subsequent locations (vectorized)
    u[i, 2:n_occasion, 1] ~ normal(u[i, 1:Kminus1, 1], sigma_move);
    u[i, 2:n_occasion, 2] ~ normal(u[i, 1:Kminus1, 2], sigma_move);
  }

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
