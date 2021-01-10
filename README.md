Here the susceptible-infected-removed (SIR) model is fitted to mortality data for the UK COVID-19 epidemic.

The coupled SIR ODEs are solved using odeint, from the scipy package. Model input parameters, including transmission rate (b), and rate of vaccination, vary over time. To avoid solver errors due to these discontinuities, odeint is used to solve for a series of intervals, end conditions for each stage being used as the initial conditions for the next.

Mortality is crudely estimated at 1% of the removed fraction of the population, with a 14 day lag assumed. The transmission rate is set by hand at key dates (e.g. in particular at onset of lockdowns) to fit against the historical mortality curve.

Findings seem to suggest that in the U,K the third lockdown, combined with a sufficiently rapid vaccination program, should be sufficient to avoid a follow on major fourth wave.

Please note that I am not an epidemiologist, and it is possible that this use of the SIR model is not strictly correct. This model is developed to aid my own understanding of the epidemic, but hopefully it may be interest to others. The code may be straightforwardly adapted to provide for projections for other countries.