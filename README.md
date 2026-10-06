# GBQR Extensions

This repository contains extensions to the epiENGAGE-GBQR forecasting framework.

The current work focuses on two extensions:

1. **Quantile-to-trajectory forecasting**  
   Generating joint multi-horizon forecast trajectories from GBQR marginal
   quantile forecasts using a Gaussian copula.

2. **Long-term forecasting**  
   Extending the existing GBQR framework beyond the current short-term
   forecast horizons.

## Repository structure

### `quantile-to-trajectory/`

Contains the implementation and evaluation of the Gaussian copula approach
used to convert GBQR marginal quantile forecasts into joint forecast
trajectories.

See `copula_method.qmd` for details on the methodology, implementation,
and diagnostic evaluation.

### `long-term-forecasting/`

Contains the development and evaluation of the long-term GBQR forecasting
framework.