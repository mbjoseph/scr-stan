
data {
  int<lower = 1> M;
  int<lower = 1> n_year;
  int<lower = M * n_year, upper = M * n_year> bigM;
  int<lower = 0, upper = n_year> year_id[bigM]; // 0 acts as NA
  int<lower = 1> n_trap;
  int<lower = 1> max_n_occasion;
  int<lower = 1, upper = max_n_occasion> n_occasion[n_year];
  matrix[n_trap, 2] X;
  int<lower = 0, upper = n_trap + 1> y[bigM, max_n_occasion]; // 0 acts as NA
  vector[2] xlim;
  vector[2] ylim;
  int<lower = 0, upper = 1> known_dead[bigM, max_n_occasion];
}

transformed data {
  real logM = log(M);
  int<lower = 0, upper = 1> observed[bigM];
  int<lower = 0, upper = 1> possibly_alive[bigM, max_n_occasion];

  for (i in 1:bigM) {
    observed[i] = 0;
    for (j in 1:max_n_occasion) {
      if (y[i, j] > 0) {
        // 0 represents NA (the jth occasion didn't happen)
        if (y[i, j] < (n_trap + 1)) {
          // any nonzero observation other than n_trap + 1 is a detection
          observed[i] = 1;
        }
      }
    }

    for (k in 1:max_n_occasion) {
      possibly_alive[i, k] = 1 - known_dead[i, k];
    }
  }
}

parameters {
  real alpha0;
  real<lower = 0> alpha1;
  vector[n_year] beta0;
  vector<lower = xlim[1], upper = xlim[2]>[bigM] s1;
  vector<lower = ylim[1], upper = ylim[2]>[bigM] s2;
}

transformed parameters {
  vector[bigM] lp_if_present;
  vector[bigM] log_lik;
  real<upper = 0> log_psi = log_sum_exp(beta0) - logM;
  real<upper = 0> log1m_psi = log1m_exp(log_psi);
  vector<upper = 0>[n_year] lp_year = log_softmax(beta0);
  vector[n_year] year_lp_vec[bigM];
  
  {
    vector[2] s;
    vector[n_trap] dist;
    vector[n_trap + 1] log_odds;
    vector[max_n_occasion] tmp;

    for (i in 1:bigM) {
      s[1] = s1[i];
      s[2] = s2[i];
      for (j in 1:n_trap) {
        dist[j] = distance(s, X[j, ]);
        log_odds[j] = alpha0 - alpha1 * dist[j];
      }
      log_odds[n_trap + 1] = 0;

      // Looping over the number occasions in each year deals with the fact
      // that in year 1, we only have 9 occasions, and in the rest 10 occasions.
      // But, this is only valid for observed individuals.
      tmp = rep_vector(0, max_n_occasion);
      year_lp_vec[i] = rep_vector(0, n_year);
      if (observed[i]) {
        // we know the year ID in which the individual occurred, and therefore
        // we know the number of sampling occasions to loop over
        for (k in 1:n_occasion[year_id[i]]) {
          if (possibly_alive[i, k]) {
            tmp[k] = categorical_logit_lpmf(y[i, k] | log_odds);
          }
        }
        lp_if_present[i] = log_psi 
                           + categorical_logit_lpmf(year_id[i] | lp_year)
                           + sum(tmp);
      } else {
        // This individual hasn't been observed.
        // We don't know the year ID, but we can marginalize over the groups
        // [y | g=1, z=1] [g = 1] + ... + [y | g=n_year, z=1] [g=n_year]
        for (j in 1:n_year) {
          year_lp_vec[i, j] = categorical_logit_lpmf(j | lp_year)
            + categorical_logit_lpmf(y[i, 1:n_occasion[j]] | log_odds);
        }
        lp_if_present[i] = log_psi + log_sum_exp(year_lp_vec[i, ]);
      }

      if (observed[i]) {
        log_lik[i] = lp_if_present[i];
      } else {
        log_lik[i] = log_sum_exp(lp_if_present[i], log1m_psi);
      }
    }
  } // end temp scope
}

model {
  // priors
  alpha0 ~ normal(0, 3);
  alpha1 ~ normal(0, 3);
  beta0 ~ normal(0, 10);
  
  // likelihood
  target += sum(log_lik);
}


generated quantities {
  int N[n_year];
  
  {
    vector[bigM] lp_present;
    int z[bigM];
    int c[bigM];

    for (i in 1:bigM) {
      if(observed[i]) {
        z[i] = 1;
        c[i] = year_id[i];
      } else {
        // [z=1][y=0 | z=1] / [y=0] on a log scale
        lp_present[i] = lp_if_present[i] - log_lik[i];
        z[i] = bernoulli_rng(exp(lp_present[i]));
        // Category probabilities are: 
        //   [C = c | y] [C = c] / [y]
        // = [C = c | y] [C = c] / \sum_{c} [C = c | y] [C = c]
        c[i] = categorical_rng(softmax(year_lp_vec[i, ]));
      }
    }
    
    // count the number of live birbs in each year
    for (t in 1:n_year) {
      N[t] = 0;
      for (i in 1:bigM) {
        if (c[i] == t) {
          N[t] += z[i];
        }
      }
    }
  }
}
