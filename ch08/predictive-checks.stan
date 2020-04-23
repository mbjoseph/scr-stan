
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
  
  // grid data for predictive checks
  int<lower = 1> n_cell;
  vector[n_cell] cell_xmin;
  vector[n_cell] cell_xmax;
  vector[n_cell] cell_ymin;
  vector[n_cell] cell_ymax;
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
  matrix[M, n_trap] logit_p;
  
  {
    matrix[M, n_trap] sq_dist;
    matrix[M, n_trap] log_p;

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
  vector[n_cell] Ng;
  
  // goodness of fit stats for uniformity/spatial randomness (pg. 234-235)
  real freeman_tukey_obs;
  real freeman_tukey_sim;
  real index_dispersion_obs;
  real index_dispersion_sim;
  
  // goodness of fit stats for the observation model (pg. 237-238)
  // "Fit statistic 1: individual x trap frequencies"
  real T1_obs;
  real T1_sim;
  // "Fit statistic 2: individual encounter histories"
  real T2_obs;
  real T2_sim;
  // "Fit statistic 3: trap frequencies"
  real T3_obs;
  real T3_sim;

  { // begin temporary scope
    vector[M] lp_present; // [z=1][y=0 | z=1] / [y=0] on a log scale
    int z[M];
    vector[M] s1_sim;
    vector[M] s2_sim;
    vector[n_cell] Ng_sim;
    matrix[M, n_trap] p;
    vector[M] T1_sums_obs;
    vector[M] T1_sums_sim;
    vector[M] T2_sums_obs;
    vector[M] T2_sums_sim;
    vector[n_trap] T3_sums_obs;
    vector[n_trap] T3_sums_sim;
    int y_sim[M, n_trap];
    matrix[M, n_trap] pK;

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
    
    for (i in 1:M) {
      if (z[i] == 1) {
        // new activity centers from predictive distribution
        s1_sim[i] = uniform_rng(xlim[1], xlim[2]);
        s2_sim[i] = uniform_rng(ylim[1], ylim[2]);
      }
      p[i, ] = z[i] * inv_logit(logit_p[i, ]);
      for (j in 1:n_trap) {
        // new detection data from the predictive distribution
        y_sim[i, j] = binomial_rng(n_occasion[j], p[i, j]);
        pK[i, j] = p[i, j] * n_occasion[j];
      }
      T1_sums_obs[i] = sum(square(sqrt(to_row_vector(y[i, ])) - sqrt(pK[i, ])));
      T1_sums_sim[i] = sum(square(sqrt(to_row_vector(y_sim[i, ])) - sqrt(pK[i, ])));
      T2_sums_obs[i] = square(sqrt(sum(y[i, ])) - sqrt(sum(pK[i, ])));
      T2_sums_sim[i] = square(sqrt(sum(y_sim[i, ])) - sqrt(sum(pK[i, ])));
    }
    
    for (j in 1:n_trap) {
      T3_sums_obs[j] = square(sqrt(sum(y[, j])) - sqrt(sum(pK[, j])));
      T3_sums_sim[j] = square(sqrt(sum(y_sim[, j])) - sqrt(sum(pK[, j])));
    }
    
    // compute the number of activity centers in each grid cell
    for (c in 1:n_cell) {
      Ng[c] = 0;
      Ng_sim[c] = 0;
      for (i in 1:M) {
        if (z[i] == 1) {
          // estimated activity centers
          if ((s1[i] >= cell_xmin[c]) * (s1[i] < cell_xmax[c])) {
            if ((s2[i] >= cell_ymin[c]) * (s2[i] < cell_ymax[c])) {
              Ng[c] += 1;
            }
          }
          // simulated activity centers (under spatial randomness)
          if ((s1_sim[i] >= cell_xmin[c]) * (s1_sim[i] < cell_xmax[c])) {
            if ((s2_sim[i] >= cell_ymin[c]) * (s2_sim[i] < cell_ymax[c])) {
              Ng_sim[c] += 1;
            }
          }
        }
      }
    }
    freeman_tukey_obs = sum(square(sqrt(Ng) - sqrt(mean(Ng))));
    freeman_tukey_sim = sum(square(sqrt(Ng_sim) - sqrt(mean(Ng_sim))));
    index_dispersion_obs = variance(Ng) / mean(Ng);
    index_dispersion_sim = variance(Ng_sim) / mean(Ng_sim);
    T1_obs = sum(T1_sums_obs);
    T1_sim = sum(T1_sums_sim);
    T2_obs = sum(T2_sums_obs);
    T2_sim = sum(T2_sums_sim);
    T3_obs = sum(T3_sums_obs);
    T3_sim = sum(T3_sums_sim);
  } // end temporary scope
}
