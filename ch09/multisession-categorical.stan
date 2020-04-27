
data {
  int<lower = 1> M;
  int<lower = 1> n_year;
  int<lower = M * n_year, upper = M * n_year> bigM;
  int<lower = 1, upper = n_year> year[bigM];
  int<lower = 1> n_trap;
  int<lower = 1> max_n_occasion;
  int<lower = 1, upper = max_n_occasion> n_occasion[n_year];
  matrix[n_trap, 2] X;
  int<lower = 0, upper = n_trap + 1> y[bigM, max_n_occasion]; // 0 acts as NA
  int<lower = 0, upper = 1> known_dead[bigM, max_n_occasion];
  vector[2] xlim;
  vector[2] ylim;
}

transformed data {
  int<lower = 0, upper = 1> observed[bigM];
  int<lower = 0, upper = 1> not_known_dead[bigM, max_n_occasion];

  for (i in 1:bigM) {
    observed[i] = 0;
    for (j in 1:n_occasion[year[i]]) {
      if (y[i, j] < (n_trap) + 1) {
        observed[i] = 1;
      }
    }

    for (k in 1:max_n_occasion) {
      not_known_dead[i, k] = 1 - known_dead[i, k];
    }
  }
}

parameters {
  real alpha0;
  real<lower = 0> alpha1;
  vector<lower = 0, upper = 1>[n_year] psi;
  vector<lower = xlim[1], upper = xlim[2]>[bigM] s1;
  vector<lower = ylim[1], upper = ylim[2]>[bigM] s2;
}

transformed parameters {
  vector[bigM] lp_if_present;
  vector[bigM] log_lik;
  
  {
    matrix[bigM, 2] s = append_col(s1, s2);
    matrix[bigM, n_trap] dist;
    matrix[max_n_occasion, n_trap + 1] logits;
    vector[max_n_occasion] tmp;

    for (i in 1:bigM) {
      for (j in 1:n_trap) {
        dist[i, j] = distance(s[i, ], X[j, ]);
        logits[, j] = rep_vector(alpha0 - alpha1 * dist[i, j], 
                                      max_n_occasion);
      }
      logits[, n_trap + 1] = rep_vector(0, max_n_occasion);

      // Looping over the number occasions in each year deals with the fact
      // that in year 1, we only have 9 occasions, and in the rest 10 occasions.
      tmp = rep_vector(0, max_n_occasion);
      for (k in 1:n_occasion[year[i]]) {
        if (not_known_dead[i, k]) { // data from the dead doesn't mean anything
          tmp[k] = categorical_logit_lpmf(y[i, k] | to_vector(logits[k,]));
        }
      }
      lp_if_present[i] = bernoulli_lpmf(1 | psi[year[i]]) + sum(tmp);
      
      if (observed[i]) {
        log_lik[i] = lp_if_present[i];
      } else {
        log_lik[i] = log_sum_exp(lp_if_present[i], 
                                 bernoulli_lpmf(0 | psi[year[i]]));
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
  int N[n_year];
  
  {
    vector[bigM] lp_present; // [z=1][y=0 | z=1] / [y=0] on a log scale
    int z[bigM];

    for (i in 1:bigM) {
      if(observed[i]) {
        z[i] = 1;
      } else {
        lp_present[i] = lp_if_present[i] - log_lik[i];
        z[i] = bernoulli_rng(exp(lp_present[i]));
      }
    }
    
    // count the number of live birbs in each year
    for (t in 1:n_year) {
      N[t] = 0;
      for (i in 1:bigM) {
        if (year[i] == t) {
          N[t] += z[i];
        }
      }
    }
  }
}
