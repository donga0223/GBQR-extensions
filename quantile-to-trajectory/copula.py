import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import minimize

def calculate_pit(group):
    g = group.sort_values("output_type_id")

    probs = g["output_type_id"].astype(float).to_numpy()
    qvals = g["value"].astype(float).to_numpy()
    y = float(g["oracle_value"].iloc[0])

    z_grid = norm.ppf(probs)

    # Interior: piecewise-linear interpolation on probit scale
    if qvals[0] <= y <= qvals[-1]:
        z = np.interp(y, qvals, z_grid)

    # Lower tail: regression using .025, .05, .10
    elif y < qvals[0]:
        idx = np.isin(probs, [0.025, 0.05, 0.10])

        slope, intercept = np.polyfit(
            z_grid[idx],
            qvals[idx],
            1
        )

        z = (y - intercept) / slope

    # Upper tail: regression using .90, .95, .975
    else:
        idx = np.isin(probs, [0.90, 0.95, 0.975])

        slope, intercept = np.polyfit(
            z_grid[idx],
            qvals[idx],
            1
        )

        z = (y - intercept) / slope

    # Bound extreme extrapolated z values
    eps = 1e-7
    z_bound = norm.ppf(1 - eps)

    z = np.clip(z, -z_bound, z_bound)

    pit = norm.cdf(z)
        
    return pd.Series({
        "pit": pit,
        "z": z,
        "oracle_value": y
    })



def make_sigma(xi):
    xi = np.asarray(xi, dtype=float)

    n_horizons = len(xi) + 1
    Sigma = np.eye(n_horizons)

    for lag, rho in enumerate(xi, start=1):
        i = np.arange(n_horizons - lag)
        Sigma[i, i + lag] = rho
        Sigma[i + lag, i] = rho

    return Sigma

def neg_log_copula_likelihood(xi, Z):
    Sigma = make_sigma(xi)

    # Sigma must be positive definite
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return 1e10

    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return 1e10

    A = Sigma_inv - np.eye(4)

    quadratic = np.einsum(
        "ni,ij,nj->n",
        Z, A, Z
    )

    log_copula = -0.5 * logdet - 0.5 * quadratic

    return -log_copula.sum()


def z_to_forecast(z, probs, qvals):
    probs = np.asarray(probs, dtype=float)
    qvals = np.asarray(qvals, dtype=float)

    z_grid = norm.ppf(probs)

    # Interior
    if z_grid[0] <= z <= z_grid[-1]:
        return np.interp(z, z_grid, qvals)

    # Lower tail: regression using 0.025, 0.05, 0.10
    elif z < z_grid[0]:
        idx = np.isin(probs, [0.025, 0.05, 0.10])
        slope, intercept = np.polyfit(
            z_grid[idx],
            qvals[idx],
            1
        )
        return intercept + slope * z

    # Upper tail: regression using 0.90, 0.95, 0.975
    else:
        idx = np.isin(probs, [0.90, 0.95, 0.975])
        slope, intercept = np.polyfit(
            z_grid[idx],
            qvals[idx],
            1
        )
        return intercept + slope * z

def fit_copula(Z):
    n_horizons = Z.shape[1]

    fit = minimize(
        neg_log_copula_likelihood,
        x0=np.repeat(0.3, n_horizons - 1),
        args=(Z,),
        method="L-BFGS-B",
        bounds=[(-0.99, 0.99)] * (n_horizons - 1)
    )

    if not fit.success:
        raise RuntimeError(
            f"Copula optimization failed: {fit.message}"
        )

    xi_hat = fit.x
    Sigma_hat = make_sigma(xi_hat)

    return Sigma_hat, xi_hat, fit


def generate_copula_trajectories(
    forecast,
    Sigma,
    n_trajectories=100,
    seed=123
):
    rng = np.random.default_rng(seed)

    horizons = np.sort(forecast["horizon"].unique())
    n_horizons = len(horizons)

    if Sigma.shape != (n_horizons, n_horizons):
        raise ValueError(
            "Sigma dimension does not match number of horizons."
        )

    Z_sim = rng.multivariate_normal(
        mean=np.zeros(n_horizons),
        cov=Sigma,
        size=n_trajectories
    )

    Y_sim = np.zeros_like(Z_sim)

    for j, h in enumerate(horizons):
        fcst_h = (
            forecast[forecast["horizon"] == h]
            .sort_values("output_type_id")
        )

        probs = fcst_h["output_type_id"].astype(float).to_numpy()
        qvals = fcst_h["value"].astype(float).to_numpy()

        Y_sim[:, j] = [
            z_to_forecast(z, probs, qvals)
            for z in Z_sim[:, j]
        ]

    # influenza ED percentage cannot be negative
    Y_sim = np.maximum(Y_sim, 0)

    return Y_sim

def generate_trajectories_for_date(
    forecasts,
    reference_date,
    Sigma,
    n_trajectories=100,
    seed=123
):
    reference_date = pd.to_datetime(reference_date)

    fcst_date = forecasts[
        pd.to_datetime(forecasts["reference_date"]) == reference_date
    ].copy()

    locations = sorted(fcst_date["location"].unique())

    results = []

    for loc_idx, location in enumerate(locations):

        fcst_loc = fcst_date[
            fcst_date["location"] == location
        ].copy()

        horizons = sorted(fcst_loc["horizon"].unique())

        # Skip incomplete forecasts
        if len(horizons) != Sigma.shape[0]:
            continue

        Y_sim = generate_copula_trajectories(
            forecast=fcst_loc,
            Sigma=Sigma,
            n_trajectories=n_trajectories,
            seed=seed + loc_idx
        )

        target_dates = (
            fcst_loc[
                ["horizon", "target_end_date"]
            ]
            .drop_duplicates()
            .sort_values("horizon")
        )

        for trajectory_id in range(n_trajectories):

            temp = target_dates.copy()

            temp["reference_date"] = reference_date
            temp["location"] = location
            temp["trajectory_id"] = trajectory_id
            temp["value"] = Y_sim[trajectory_id, :]

            results.append(temp)

    trajectories = pd.concat(
        results,
        ignore_index=True
    )

    return trajectories


def plot_all_copula_trajectories(
    forecasts,
    observed,
    trajectories,
    reference_date,
    ncols=5,
    figsize_per_panel=(4, 3),
    max_trajectories=100,
    weeks_before=1,
    weeks_after=6
):
    reference_date = pd.to_datetime(reference_date)

    # Forecasts for selected reference date
    fcst_date = forecasts[
        pd.to_datetime(forecasts["reference_date"]) == reference_date
    ].copy()

    # Trajectories for selected reference date
    traj_date = trajectories[
        pd.to_datetime(trajectories["reference_date"]) == reference_date
    ].copy()

    locations = sorted(traj_date["location"].unique())

    # Plot window
    forecast_end = pd.to_datetime(
        fcst_date["target_end_date"]
    ).max()

    # Plot window
    plot_start = pd.to_datetime(observed["target_end_date"]).min()

    forecast_end = pd.to_datetime(
        fcst_date["target_end_date"]
    ).max()

    plot_end = forecast_end + pd.Timedelta(weeks=weeks_after)

    nrows = int(np.ceil(len(locations) / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_panel[0] * ncols,
            figsize_per_panel[1] * nrows
        ),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, location in zip(axes, locations):

        fcst_loc = fcst_date[
            fcst_date["location"] == location
        ].copy()

        target = fcst_loc["target"].iloc[0]

        target_dates = (
            fcst_loc[
                ["horizon", "target_end_date"]
            ]
            .drop_duplicates()
            .sort_values("horizon")["target_end_date"]
            .to_numpy()
        )

        # 95% PI
        lower95 = (
            fcst_loc[
                fcst_loc["output_type_id"].astype(float) == 0.025
            ]
            .sort_values("horizon")["value"]
            .to_numpy()
        )

        upper95 = (
            fcst_loc[
                fcst_loc["output_type_id"].astype(float) == 0.975
            ]
            .sort_values("horizon")["value"]
            .to_numpy()
        )

        # GBQR median
        median = (
            fcst_loc[
                fcst_loc["output_type_id"].astype(float) == 0.5
            ]
            .sort_values("horizon")["value"]
            .to_numpy()
        )

        # Observed
        obs_loc = (
            observed[
                (observed["location"] == location) &
                (observed["target"] == target)
            ]
            .sort_values("target_end_date")
        )

        # Copula trajectories
        traj_loc = traj_date[
            traj_date["location"] == location
        ]

        trajectory_ids = (
            traj_loc["trajectory_id"]
            .drop_duplicates()
            .sort_values()
            .iloc[:max_trajectories]
        )

        for trajectory_id in trajectory_ids:

            temp = (
                traj_loc[
                    traj_loc["trajectory_id"] == trajectory_id
                ]
                .sort_values("horizon")
            )

            ax.plot(
                temp["target_end_date"],
                temp["value"],
                linewidth=0.7,
                alpha=0.25
            )

        # GBQR 95% PI
        ax.fill_between(
            target_dates,
            lower95,
            upper95,
            alpha=0.20
        )

        # GBQR median
        ax.plot(
            target_dates,
            median,
            linewidth=2
        )

        # Observed
        ax.plot(
            obs_loc["target_end_date"],
            obs_loc["oracle_value"],
            linewidth=2
        )

        # Reference date
        ax.axvline(
            reference_date,
            linestyle="--",
            linewidth=1
        )

        # Limit displayed time window
        ax.set_xlim(plot_start, plot_end)

        ax.set_title(location, fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    # Remove unused panels
    for ax in axes[len(locations):]:
        ax.remove()

    fig.suptitle(
        f"Copula trajectories — reference date {reference_date.date()}",
        fontsize=16,
        y=1.002
    )

    fig.supxlabel("Date")
    fig.supylabel("% ED visits due to influenza")

    plt.tight_layout()
    plt.show()
