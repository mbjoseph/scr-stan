// Cormack-Jolly-Seber model as a multistate hidden Markov model
// adapted from stan-dev/example-models/blob/master/BPA/Ch.10/js_ms.stan
//--------------------------------------
// States:
// 1 not captured yet
// 2 alive
// 3 dead
// Observations: 
// 1 detected
// 2 not detected

data {
  int<lower = 0> M;
  int<lower = 0> T;
  int<lower = 0, upper = 2> y[M, T];
  int<lower = 1, upper = T> first_capture[M];
}

transformed data {
  int Tm1 = T - 1;
}

parameters {
  real<lower = 0, upper = 1> phi;   // survival
  vector<lower = 0, upper = 1>[Tm1] p;  // detection
  // note that we have a vector of length Tm1 for detection probability, 
  // because the CJS model is conditional on first detection. Therefore, 
  // there is no detection probability associated with the first time period.
}

transformed parameters {
  vector[M] log_lik;

  { // begin local scope
    real c_phi = 1 - phi;
    vector[Tm1] c_p = 1 - p;
    real acc[3];
    vector[3] gam[T];
    vector[3] ps[3, T];
    vector[2] po[3, Tm1];
    
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

      // fill in observation probability matrix
      for (t in 1:Tm1) {
        po[1, t, 1] = 0;         // untracked individuals (s=1) not detected
        po[1, t, 2] = 1;
        po[2, t, 1] = p[t];      // live individuals (s=2) detected with pr=p
        po[2, t, 2] = c_p[t];
        po[3, t, 1] = 0;         // dead individuals (s=3) not detected
        po[3, t, 2] = 1;
      }
    }
    
    for (i in 1:M) {
      // All individuals are in state 1 prior to entering
      gam[first_capture[i], 1] = 0;
      gam[first_capture[i], 2] = 1;
      gam[first_capture[i], 3] = 0;
  
      for (t in (first_capture[i] + 1):T) {
        for (k in 1:3) {
          for (j in 1:3) {
            acc[j] = gam[t - 1, j] * ps[j, t - 1, k] * po[k, t-1, y[i, t]];
          }
          gam[t, k] = sum(acc);
        }
      }
      log_lik[i] = log(sum(gam[T]));
    }
  }
}

model {
  target += sum(log_lik);
}

// note that we could sample z and N in the generated quantities block
