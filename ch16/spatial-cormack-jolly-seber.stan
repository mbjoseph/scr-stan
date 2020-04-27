// Cormack-Jolly-Seber model as a multistate hidden Markov model
// adapted from stan-dev/example-models/blob/master/BPA/Ch.10/js_ms.stan
//--------------------------------------
// States:
// 1 not captured yet
// 2 alive
// 3 dead
// Observations (Poisson model): 
// 1: 0 count
// 2: 1 count
// ...
// n_max + 1: max. observed count
// n_max + 2: counts greater than observed

data {
  int<lower = 0> M;
  int<lower = 0> T;
  int<lower = 1> ntrap;
  real xlo;
  real xhi;
  vector<lower = xlo, upper = xhi>[ntrap] trap_x;
  int<lower = 0> ymax;
  int<lower = 0, upper = ymax> y[M, ntrap, T];
  int<lower = 1, upper = T> first_capture[M];
}

transformed data {
  int Tm1 = T - 1;
}

parameters {
  real<lower = 0, upper = 1> phi;   // survival
  // note: in the book, s varies by individual and time, but here it varies
  // only by individual to keep the run times manageable. To make it vary by
  // time as well, you could declare an array of vectors instead of a vector.
  vector<lower = xlo, upper = xhi>[M] s;
  real p0;
  real<lower = 0> alpha1;
}

transformed parameters {
  vector[M] log_lik;
  real log_p0 = log(p0);
  
  { // begin local scope
    real c_phi = 1 - phi;
    real acc[3];
    real loglam;
    vector[3] gam[T];
    vector[3] ps[3, T];
    vector[ymax + 2] po[3, ntrap];
  
    for (i in 1:M) {
      // fill in state transition probability matrix
      for (t in 1:T) {
          ps[1, t, 1] = 1;
          ps[1, t, 2] = 0;
          ps[1, t, 3] = 0;
          ps[2, t, 1] = 0;
          ps[2, t, 2] = phi;
          ps[2, t, 3] = c_phi;
          ps[3, t, 1] = 0;
          ps[3, t, 2] = 0;
          ps[3, t, 3] = 1;
      }

      for (trap in 1:ntrap) {
        loglam = log_p0 - alpha1*(s[i] - trap_x[trap])^2;
        // observation probabilities for zero counts
        po[1, trap, 1] = 1; 
        po[2, trap, 1] = exp(poisson_log_lpmf(0 | loglam));
        po[3, trap, 1] = 1;
        // observation probabilities for counts from 1:ymax
        for (j in 1:ymax) {
          po[1, trap, j+1] = 0;
          po[2, trap, j+1] = exp(poisson_log_lpmf(j | loglam));
          po[3, trap, j+1] = 0;
        }
        // observation probabilities for counts > ymax
        po[1, trap, ymax + 2] = 0; 
        po[2, trap, ymax + 2] = 1 - sum(po[2, trap, 1:(ymax + 1)]);
        po[3, trap, ymax + 2] = 0;
      }

      // All individuals are in state 1 prior to entering
      gam[first_capture[i], 1] = 0;
      gam[first_capture[i], 2] = 1;
      gam[first_capture[i], 3] = 0;
  
      for (t in (first_capture[i] + 1):T) {
        for (k in 1:3) {
          for (j in 1:3) {
            acc[j] = gam[t - 1, j] * ps[j, t - 1, k];
            for (trap in 1:ntrap) {
              acc[j] *= po[k, trap, y[i, trap, t] + 1];
            }
          }
          gam[t, k] = sum(acc);
        }
      }
      log_lik[i] = log(sum(gam[T]));
    }
  }
}

model {
  p0 ~ normal(0, 3);
  alpha1 ~ normal(0, 3);
  target += sum(log_lik);
}

// note that we could sample z and N in the generated quantities block
