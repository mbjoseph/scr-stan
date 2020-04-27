# Chapter 5: fully spatial capture-recapture models

These are the simplest SCR models described in the book. 

### Known N (SCR0)

See [scr0.R](scr0.R).

### Unknown N (SCR0)

See [scr0-data-augmentation.R](scr0-data-augmentation.R), which uses data 
augmentation to estimate N. The Stan implementation marginalizes over `z`, 
then samples from `z` and `N` in the generated quantities block. 
