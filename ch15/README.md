# Chapter 15: models for search-encounter data

These are more computationally challenging than many of the previous
models. There may be opportunities to reparameterize. 

1. **Fixed search path.** See
[fixed-search-path.R](fixed-search-path.R), which models detections along a 
fixed search path, and allows the instantaneous locations of animals to vary
by occasion around individual-specific activity centers.

2. **Uniform search intensity.** See 
[rw-uniform-search.R](rw-uniform-search.R) for an example where individual
locations are modeled as evolving through time by random walks. 
