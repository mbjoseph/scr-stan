// Jolly-Seber model as a multistate hidden Markov model
// adapted from stan-dev/example-models/blob/master/BPA/Ch.10/js_ms.stan
//--------------------------------------
// States:
// 1 not yet entered
// 2 alive
// 3 dead

data {
  int<lower = 0> M;
  int<lower = 0> T;
  int<lower = 0> K[T];
  int<lower = 0> Kmax;
  int<lower = 1> n_trap;
  int<lower = 0, upper = n_trap + 1> y[M, Kmax, T];
  matrix[n_trap, 2] X;
  vector[2] xlim;
  vector[2] ylim;
}

transformed data {
  int Tm1 = T - 1;
  int Tp1 = T + 1;
}

parameters {
  vector<lower = 0,upper = 1>[Tm1] gamma;         // recruitment
  real<lower = 0, upper = 1> psi;                 // initial pr. of being alive
  real<lower = 0, upper = 1> phi;                 // survival
  vector<lower = xlim[1], upper = xlim[2]>[T] sx[M];
  vector<lower = ylim[1], upper = ylim[2]>[T] sy[M];
  real p0;
  real<lower = 0> alpha1;
}

transformed parameters {
  real<lower = 0, upper = 1> c_phi = 1 - phi;
  real log_p0 = log(p0);
  simplex[3] ps[3, T];
  vector[M] log_lik;

  // Define probabilities of state S(t+1) given S(t)
  ps[1, 1, 1] = 1 - psi;
  ps[1, 1, 2] = psi;
  ps[1, 1, 3] = 0;
  ps[2, 1, 1] = 0;
  ps[2, 1, 2] = 1;
  ps[2, 1, 3] = 0;
  ps[3, 1, 1] = 0;
  ps[3, 1, 2] = 0;
  ps[3, 1, 3] = 1;
  
  for (t in 2:T) {
      ps[1, t, 1] = 1 - gamma[t-1];
      ps[1, t, 2] = gamma[t-1];
      ps[1, t, 3] = 0;
      ps[2, t, 1] = 0;
      ps[2, t, 2] = phi;
      ps[2, t, 3] = c_phi;
      ps[3, t, 1] = 0;
      ps[3, t, 2] = 0;
      ps[3, t, 3] = 1;
  }

  { // begin local scope
    vector[2] s;
    vector[n_trap + 1] logits;
    real acc[3];
    vector[3] gam[Tp1];
    vector[n_trap + 1] po[T, 3];

    for (i in 1:M) {
      for (t in 1:T) {
        s[1] = sx[i, t];
        s[2] = sy[i, t];
        for (j in 1:n_trap) {
          po[t, 1, j] = 0; // not recruited
          po[t, 3, j] = 0; // dead
          logits[j] = log_p0 - alpha1 * squared_distance(s, X[j, ]);
        }
        logits[n_trap + 1] = 0;
        po[t, 1, n_trap + 1] = 1;
        po[t, 3, n_trap + 1] = 1;
        po[t, 2, ] = softmax(logits);
      }
    }
    
    for (i in 1:M) {
      // All individuals are in state 1 (not recruited) at t=0
      gam[1, 1] = 1;
      gam[1, 2] = 0;
      gam[1, 3] = 0;
  
      // we iterate to T + 1, because we inserted a dummy period where 
      // every individual is in the "not recruited" state
      for (t in 2:(Tp1)) {
        for (k in 1:3) {
          for (j in 1:3) {
            acc[j] = gam[t - 1, j] * ps[j, t - 1, k];
            // the loop below accounts for differences in K across years
            for (occasion in 1:K[t-1]) {
              acc[j] *= po[t-1, k, y[i, occasion, t - 1]];
            }
          }
          gam[t, k] = sum(acc);
        }
      }
      log_lik[i] = log(sum(gam[Tp1]));
    }
  }
}

model {
  p0 ~ normal(0, 3);
  alpha1 ~ lognormal(0, 1);
  target += sum(log_lik);
}

// note that we could sample z and N in the generated quantities block
