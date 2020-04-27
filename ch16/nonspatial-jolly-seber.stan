// Jolly-Seber model as a multistate hidden Markov model
// adapted from stan-dev/example-models/blob/master/BPA/Ch.10/js_ms.stan
//--------------------------------------
// States:
// 1 not yet entered
// 2 alive
// 3 dead
// Observations: 
// 1 detected
// 2 not detected
// 0 NA (no sampling occasion took place)

data {
  int<lower = 0> M;
  int<lower = 0> T;
  int<lower = 0> K[T];
  int<lower = 0> Kmax;
  int<lower = 0, upper = 2> y[M, Kmax, T];
}

transformed data {
  int Tm1 = T - 1;
  int Tp1 = T + 1;
}

parameters {
  vector<lower = 0,upper = 1>[Tm1] gamma;  // recruitment
  real<lower = 0, upper = 1> psi;          // initial pr. of being alive
  real<lower = 0, upper = 1> phi;          // survival
  real<lower = 0, upper = 1> p;            // detection
}

transformed parameters {
  real<lower = 0, upper = 1> c_phi = 1 - phi;
  simplex[3] ps[3, T];
  simplex[2] po[3];

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

  // Define probabilities of O(t) given S(t)
  po[1, 1] = 0;        // not recruited individuals (s=1) are never detected
  po[1, 2] = 1;
  po[2, 1] = p;         // live individuals (s=2) detected with prob. p
  po[2, 2] = 1 - p;
  po[3, 1] = 0;        // dead individuals (s=3) not detected
  po[3, 2] = 1;
}

model {
  real acc[3];
  vector[3] gam[Tp1];

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
            acc[j] *= po[k, y[i, occasion, t - 1]];
          }
        }
        gam[t, k] = sum(acc);
      }
    }
    target += log(sum(gam[Tp1]));
  }
}

// note that we could sample z and N in the generated quantities block
