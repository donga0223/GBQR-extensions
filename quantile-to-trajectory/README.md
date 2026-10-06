# Generating Forecast Trajectories from GBQR Quantile Forecasts

**Gaussian copula extension for epiENGAGE-GBQR**

## Overview

The epiENGAGE-GBQR model produces marginal probabilistic forecasts separately for each forecast horizon. For Flu-MetroCast, the short-term model produces quantile forecasts for horizons 0--3. These marginal forecasts characterize uncertainty at each horizon, but they do not specify the joint dependence among horizons and therefore cannot directly be interpreted as forecast trajectories.

This extension uses a **Gaussian copula** to convert the marginal GBQR quantile forecasts into joint multi-horizon trajectories. The goal is to preserve the original GBQR marginal predictive distributions while adding an estimated dependence structure across forecast horizons.

The workflow is:

1. Match historical GBQR forecasts to observations.
2. Calculate probability integral transform (PIT) values for the observations under the GBQR marginal distributions.
3. Transform PIT values to latent normal scores.
4. Estimate a Gaussian copula with a Toeplitz correlation structure.
5. Draw correlated latent-normal trajectories.
6. Transform each draw back through the corresponding GBQR marginal distribution.
7. Save the resulting trajectories by reference date and location.

The implementation is based on the general copula-estimation approach used in the accompanying `copula-estimation-step.R`, adapted here to quantile-based GBQR forecasts rather than full KCDE predictive distributions. In the reference implementation, PIT trajectories are assembled across prediction horizons and a Gaussian copula with a Toeplitz dependence structure is estimated by maximum likelihood using L-BFGS-B.

## Setup

The reusable copula functions are stored in `copula.py`.

```python
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

from copula import (
    calculate_pit,
    fit_copula,
    generate_copula_trajectories,
    generate_trajectories_for_date
)
```

The examples below assume that the Flu-MetroCast GBQR forecasts have been loaded into `gbqr` and the corresponding observed values into `observed`.

## Marginal GBQR forecasts

For each combination of reference date, location, and forecast horizon, GBQR provides quantile forecasts at

$$
p \in \{0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975\}.
$$

For the current short-term implementation, forecasts are produced for horizons 0, 1, 2, and 3.

The copula procedure does **not** replace or refit these marginal forecasts. Instead, it uses them as the marginal predictive distributions and estimates only the dependence among horizons.

## Match forecasts with observations

Historical forecasts are first matched to the corresponding observed influenza activity using location, target, and target end date.

```python
gbqr["target_end_date"] = pd.to_datetime(gbqr["target_end_date"])
observed["target_end_date"] = pd.to_datetime(observed["target_end_date"])

gbqr_obs = gbqr.merge(
    observed,
    on=["location", "target", "target_end_date"],
    how="left"
)
```

Each `(reference_date, location, horizon)` combination then contains the marginal quantile forecast and the realized observation.

## PIT calculation

For a continuous predictive distribution \(F_h\) at horizon \(h\), the PIT corresponding to the realized observation \(y_h\) is

$$
u_h = F_h(y_h).
$$

Because GBQR provides a finite set of quantiles rather than a complete CDF, \(F_h\) must be reconstructed from the quantile forecasts.

### Interpolation on the probit scale

Let \(p_k\) denote the forecast probabilities and \(q_k\) their corresponding forecast values. The probabilities are first transformed to standard-normal quantiles,

$$
z_k = \Phi^{-1}(p_k),
$$

where \(\Phi\) is the standard-normal CDF.

For an observation lying within the available GBQR quantile range, its latent score is obtained by piecewise-linear interpolation between the forecast values \(q_k\) and the corresponding \(z_k\).

This means that interpolation is performed between

$$
(q_k,\, \Phi^{-1}(p_k)),
$$

rather than directly between forecast values and probabilities.

### Tail extrapolation

An observed value can fall below the 0.025 quantile or above the 0.975 quantile. To obtain a PIT value in these cases, the tails are extrapolated linearly on the probit scale.

For the lower tail, a line

$$
q = a_L + b_L z
$$

is fitted using the 0.025, 0.05, and 0.10 quantiles. For \(y < q_{0.025}\),

$$
z = \frac{y-a_L}{b_L}.
$$

Similarly, the upper tail is fitted using the 0.90, 0.95, and 0.975 quantiles.

### Numerical bounding

Very large extrapolated latent scores can create PIT values numerically indistinguishable from 0 or 1. For estimation, latent scores are therefore bounded using

$$
\epsilon = 10^{-7},
\qquad
z_{\max} = \Phi^{-1}(1-\epsilon),
$$

and

$$
z \leftarrow \min\{\max(z,-z_{\max}),z_{\max}\}.
$$

The PIT is then

$$
u = \Phi(z).
$$

This bounding is used for the historical PIT transformation used to estimate the copula. It is **not** applied to newly simulated latent-normal values during trajectory generation.

The reference copula implementation similarly requires PIT values strictly inside the unit interval before fitting the copula.

```python
pit_df = (
    gbqr_obs
    .groupby(
        ["reference_date", "location", "horizon"],
        as_index=False
    )
    .apply(calculate_pit)
    .reset_index(drop=True)
)
```

No additional horizon-specific centering or standardization is applied to the resulting latent scores.

## Construct latent forecast trajectories

The PIT-derived latent scores are reshaped so that each row corresponds to one `(reference_date, location)` forecast origin and each column corresponds to a forecast horizon.

```python
z_wide = (
    pit_df
    .pivot(
        index=["reference_date", "location"],
        columns="horizon",
        values="z"
    )
    .reset_index()
)

z_wide.columns.name = None

horizons = sorted(pit_df["horizon"].unique())

z_complete = z_wide.dropna(subset=horizons)

Z = z_complete[horizons].to_numpy()
```

For the current four-horizon analysis, only forecast origins with PIT values available for all four horizons are used for copula estimation.

## Gaussian copula dependence model

Let

$$
\mathbf{Z}_i =
(Z_{i,0}, Z_{i,1}, Z_{i,2}, Z_{i,3})^\top
$$

denote the latent-normal scores for forecast origin \(i\).

We model

$$
\mathbf{Z}_i \sim N(\mathbf{0}, \Sigma),
$$

where \(\Sigma\) is a correlation matrix describing dependence among forecast horizons.

### Toeplitz correlation structure

A Toeplitz structure is used so that correlation depends on the separation between horizons rather than their absolute positions. For four horizons,

$$
\Sigma =
\begin{pmatrix}
1 & \rho_1 & \rho_2 & \rho_3 \\
\rho_1 & 1 & \rho_1 & \rho_2 \\
\rho_2 & \rho_1 & 1 & \rho_1 \\
\rho_3 & \rho_2 & \rho_1 & 1
\end{pmatrix}.
$$

Thus:

- \(\rho_1\) describes dependence between horizons separated by one week,
- \(\rho_2\) describes dependence between horizons separated by two weeks, and
- \(\rho_3\) describes dependence between horizons separated by three weeks.

This follows the Toeplitz Gaussian-copula structure used in the reference implementation.

The implementation in `copula.py` constructs this matrix generically so that the same function can be used for a different number of horizons.

## Copula likelihood and estimation

For a Gaussian copula, the log copula density for latent-normal vector \(\mathbf z_i\) can be written as

$$
\log c(\mathbf u_i;\Sigma)
=
-\frac{1}{2}\log|\Sigma|
-\frac{1}{2}
\mathbf z_i^\top
(\Sigma^{-1}-I)
\mathbf z_i,
$$

where

$$
\mathbf z_i =
\left[
\Phi^{-1}(u_{i1}),\ldots,\Phi^{-1}(u_{iH})
\right]^\top.
$$

The parameters of the Toeplitz correlation matrix are estimated by maximizing the sum of this log copula density over complete forecast origins.

Optimization uses **L-BFGS-B**, with each correlation parameter constrained to

$$
-0.99 \leq \rho_j \leq 0.99.
$$

Candidate matrices that are not positive definite are assigned a large objective value and are therefore excluded by the optimization.

```python
Sigma_hat, xi_hat, fit = fit_copula(Z)

print("Estimated lag correlations:", xi_hat)
print(Sigma_hat)
print("Eigenvalues:", np.linalg.eigvalsh(Sigma_hat))
```

For the current development dataset, the estimated lag correlations were approximately

$$
(\hat\rho_1,\hat\rho_2,\hat\rho_3)
=
(0.585,\;0.337,\;0.113).
$$

The resulting estimated correlation matrix was

$$
\hat\Sigma =
\begin{pmatrix}
1.000 & 0.585 & 0.337 & 0.113 \\
0.585 & 1.000 & 0.585 & 0.337 \\
0.337 & 0.585 & 1.000 & 0.585 \\
0.113 & 0.337 & 0.585 & 1.000
\end{pmatrix}.
$$

These estimates show decreasing dependence as the separation between forecast horizons increases.

## Generate forecast trajectories

Once \(\hat\Sigma\) has been estimated, trajectory generation is separated from copula estimation.

For a new GBQR forecast origin, latent trajectories are sampled as

$$
\mathbf Z^{(s)}
\sim
N(\mathbf 0,\hat\Sigma),
\qquad
s=1,\ldots,S.
$$

For the current implementation, \(S=100\) trajectories are generated for each location and reference date.

```python
Z_sim = rng.multivariate_normal(
    mean=np.zeros(len(horizons)),
    cov=Sigma_hat,
    size=100
)
```

### Transform latent draws back to GBQR marginals

Each simulated latent value \(z_h^{(s)}\) must be mapped back to the GBQR predictive distribution for the corresponding horizon.

For values inside the available quantile range, the transformation is obtained by piecewise-linear interpolation between

$$
\left(\Phi^{-1}(p_k), q_k\right).
$$

For simulated values outside the available range, the same lower- and upper-tail linear extrapolation rules used for PIT calculation are applied.

Conceptually, this implements

$$
Y_h^{(s)}
=
F_h^{-1}\left\{\Phi\left(Z_h^{(s)}\right)\right\},
$$

where \(F_h^{-1}\) is approximated from the GBQR quantile forecasts.

Unlike the historical PIT calculation, simulated \(Z\) values are **not clipped** before the inverse transformation. This allows the Gaussian copula to generate draws throughout its full support.

Because the influenza target is nonnegative, simulated forecast values below zero after tail extrapolation are truncated at zero.

```python
Y_sim = generate_copula_trajectories(
    forecast=example_fcst,
    Sigma=Sigma_hat,
    n_trajectories=100,
    seed=123
)
```

## Generate trajectories for all locations

Trajectory generation is performed separately for each location while using the same estimated cross-horizon dependence matrix.

For a given reference date:

```python
trajectories = generate_trajectories_for_date(
    forecasts=gbqr,
    reference_date=example_date,
    Sigma=Sigma_hat,
    n_trajectories=100,
    seed=123
)
```

The output contains one row for each trajectory-horizon combination:

| Column | Description |
|---|---|
| `reference_date` | Forecast reference date |
| `location` | Forecast location |
| `trajectory_id` | Identifier for the simulated trajectory |
| `horizon` | Forecast horizon |
| `target_end_date` | Target date corresponding to the horizon |
| `value` | Simulated forecast value |

With 77 locations, 100 trajectories per location, and four horizons, a complete reference date contains

$$
77 \times 100 \times 4 = 30{,}800
$$

rows.

## Generate and save all reference dates

The same procedure can be repeated across all GBQR reference dates.

```python
output_dir = os.path.join(
    project_path,
    "model_output",
    "flu_metrocast_2627test"
)

os.makedirs(output_dir, exist_ok=True)

reference_dates = sorted(
    pd.to_datetime(gbqr["reference_date"].unique())
)

for reference_date in reference_dates:

    trajectories = generate_trajectories_for_date(
        forecasts=gbqr,
        reference_date=reference_date,
        Sigma=Sigma_hat,
        n_trajectories=100,
        seed=123
    )

    date_string = reference_date.strftime("%Y-%m-%d")

    output_path = os.path.join(
        output_dir,
        f"{date_string}-UT-GBQR.parquet"
    )

    trajectories.to_parquet(
        output_path,
        index=False
    )
```

At this development stage, one parquet file is written for each reference date.

## Diagnostic evaluation

The trajectory-generation procedure should be checked both on the latent scale and after transformation back to the forecast scale.

### Positive definiteness

The eigenvalues of the estimated correlation matrix should all be positive.

```python
np.linalg.eigvalsh(Sigma_hat)
```

### Marginal preservation

The Gaussian copula is intended to change the **joint dependence structure**, not the GBQR marginal forecasts. A useful diagnostic is therefore to generate a large number of trajectories and compare their empirical quantiles at each horizon with the original GBQR quantiles.

Small discrepancies are expected when only 100 trajectories are generated because of Monte Carlo variation.

### Visual trajectory diagnostics

For each location, trajectories can be plotted together with:

- the original GBQR 95% prediction interval,
- the GBQR median forecast,
- observed influenza activity, and
- the forecast reference date.

This provides a direct visual check that the generated trajectories are consistent with the original marginal forecast while showing correlated movement across horizons.

```python
plot_all_copula_trajectories(
    forecasts=gbqr,
    observed=observed,
    trajectories=trajectories,
    reference_date=example_date
)
```

The observed series can be displayed from the beginning of the available season through several weeks beyond the current forecast horizon, while the simulated trajectories themselves cover only the forecast horizons.

## Estimation versus operational generation

It is useful to distinguish two stages of the workflow.

### Copula estimation

Historical GBQR forecasts and corresponding observations are used to:

$$
\text{GBQR forecasts + observations}
\rightarrow
\text{PIT values}
\rightarrow
\text{latent } Z
\rightarrow
\hat\Sigma.
$$

This stage requires realized observations.

### Forecast generation

Once an appropriate \(\hat\Sigma\) is available, new trajectories can be generated using only the current GBQR marginal forecasts:

$$
\text{new GBQR quantiles}
+
\hat\Sigma
\rightarrow
\text{forecast trajectories}.
$$

This distinction is important for prospective forecasting because observations that occur after a forecast reference date cannot be used to estimate the dependence structure for that forecast.

## Current development choices and limitations

The current implementation is intended as a development version of the GBQR trajectory extension.

First, the current short-term copula is estimated by pooling complete forecast origins across locations. Thus, all locations share the same estimated cross-horizon correlation structure. Location-specific or region-specific dependence structures could be investigated in future work if sufficient historical information is available.

Second, the current development analysis estimates the copula using the available 2025--26 forecast/observation pairs and then uses the resulting matrix to demonstrate trajectory generation over the same season. This is appropriate for method development and retrospective diagnostics, but it should not be interpreted as a prospective evaluation. An operational implementation must estimate or update the copula using only information available at each forecast date.

Third, the current short-term analysis uses complete four-horizon PIT trajectories. Extending the same framework to substantially longer forecast horizons creates additional challenges because later horizons have fewer complete historical forecast origins. The long-term implementation will therefore be documented separately once its dependence-estimation strategy is finalized.

## Implementation files

The main reusable functions are contained in `copula.py`:

- `calculate_pit()` -- converts observed outcomes under GBQR marginal forecasts to PIT values and latent-normal scores.
- `make_sigma()` -- constructs the Toeplitz correlation matrix.
- `neg_log_copula_likelihood()` -- evaluates the Gaussian copula objective.
- `fit_copula()` -- estimates the Toeplitz dependence parameters.
- `z_to_forecast()` -- maps latent-normal draws back to the GBQR forecast scale.
- `generate_copula_trajectories()` -- generates joint trajectories for one marginal forecast set.
- `generate_trajectories_for_date()` -- generates trajectories for all locations for one reference date.

Exploratory diagnostics and plotting code are intentionally kept outside the reusable module.
