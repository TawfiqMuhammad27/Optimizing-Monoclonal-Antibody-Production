# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

import os
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import minimize

from src.classes import StandardScalerVAR
from src.functions import generate_x_lag_cur_var, progress_bar, simulate, calculate_economics

n_simulations = 20
p_max = 2
cv_splits = 3
poly_degree_max = 2
knn_k_max = 3

prediction_interval = 0.95
alpha = 1 - prediction_interval

model_types = ["poly", "knn"]

param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(2, poly_degree_max + 1)])
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(2, knn_k_max + 1)],
                      kneighborsregressor__weights=["uniform"])

rng = np.random.default_rng(0)

# retrieve all batch csv files
batch_files = sorted(glob.glob(os.path.join("datasets", "batch_*.csv")))
batches = []
batch_ids = []
for batch_file in batch_files:
    df = pd.read_csv(batch_file)
    batch_id = int(os.path.basename(batch_file).replace("batch_", "").replace(".csv", ""))
    df["Batch ID"] = batch_id
    batch_ids.append(batch_id)
    batches.append(df)

features = batches[0].columns[1:-1]

feature_plot = "mAb [mg/L]"
idx_fp = np.argmax(features == feature_plot)

units = "[mg/L]"
df_ic = pd.read_csv(os.path.join("datasets", "initial_conditions.csv"))
ic_glucose = np.unique(df_ic["Glucose [mM]"])
ic_glutamine = np.unique(df_ic["Glutamine [mM]"])
ic_VCD = np.unique(df_ic["VCD [cells/mL]"])


nrows = len(ic_glucose)
ncols = len(ic_glutamine)

n_iters = len(model_types) * p_max * len(batch_ids) * n_simulations
c_iter = 0
records = []
df_all = pd.concat(batches)

print("Starting model evaluation loops")

for model_type in model_types:
    # Evaluate NVAR models with different numbers of lagged values
    for p in range(1, p_max + 1):
        # Create a figure with shared axes and appropriate sizing to prevent overcrowding
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 2.8, nrows * 2.2),
            dpi=300,
            sharex=True,
            sharey=True,
            layout="constrained"
        )
        axes = np.atleast_2d(axes)
        y_min = 0
        y_max = np.max(df_all.loc[:, feature_plot])
        rmse_forecasts = []

        for batch_id_test in batch_ids:
            batch_ids_train = np.array([i for i in batch_ids if i != batch_id_test])

            x_lag_train, x_cur_train, _, b_cur_train = generate_x_lag_cur_var(
                p=p, batch_ids=batch_ids_train, df=df_all, features=features, col_batch="Batch ID"
            )
            a_min = np.min(x_cur_train, axis=0)
            a_max = np.max(x_cur_train, axis=0)

            # setting up pipelines depending on what model we're running
            if model_type == "poly":
                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    PolynomialFeatures(include_bias=False),
                    LinearRegression()
                )
                param_grid = param_grid_poly
            elif model_type == "knn":
                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    KNeighborsRegressor()
                )
                param_grid = param_grid_knn
            else:
                model = None
                param_grid = {}

            gkf = GroupKFold(n_splits=cv_splits)
            model_gscv = GridSearchCV(
                model, param_grid=param_grid, cv=gkf,
                scoring="neg_root_mean_squared_error", return_train_score=True
            )

            model_gscv.fit(x_lag_train, x_cur_train, groups=b_cur_train)

            x_cur_train_pred = model_gscv.predict(x_lag_train)
            x_cur_train_res = x_cur_train - x_cur_train_pred

            idx_ic = (df_ic.index[df_ic["Batch ID"] == batch_id_test][0])


            x_initial = np.array([
                df_ic.loc[idx_ic, "Glucose [mM]"], df_ic.loc[idx_ic, "Glutamine [mM]"], 0, 0,
                df_ic.loc[idx_ic, "VCD [cells/mL]"], df_ic.loc[idx_ic, "VCD [cells/mL]"], 0
            ])

            initial_glucose = df_ic.at[idx_ic, "Glucose [mM]"]
            initial_glutamine = df_ic.at[idx_ic, "Glutamine [mM]"]
            initial_VCD = df_ic.at[idx_ic, "VCD [cells/mL]"]

            # Index matching for the 2D grid
            i_ax = np.argmax(ic_glucose == initial_glucose)
            j_ax = np.argmax(ic_glutamine == initial_glutamine)
            ax = axes[i_ax, j_ax]

            # Extract the data for the test batch
            mask = df_all.loc[:, "Batch ID"] == batch_id_test
            t_test = df_all.loc[mask, "Time [h]"].to_numpy()
            x_test = df_all.loc[mask, features].to_numpy()

            # Perform multiple forecast simulations to estimate uncertainty
            x_forecasts = []

            for _ in range(n_simulations):
                x_forecast = [x_initial]

                while len(x_forecast) < t_test.shape[0]:
                    x_lag = x_forecast[::-1][:p].copy()

                    while len(x_lag) < p:
                        x_lag.append(x_lag[-1])

                    x_lag = np.concatenate(x_lag, axis=0)
                    x_lag = np.expand_dims(x_lag, axis=0)

                    white_noise = []
                    for j in range(x_cur_train_res.shape[1]):
                        white_noise.append(rng.choice(x_cur_train_res[:, j]))
                    white_noise = np.array([white_noise])

                    x_cur_baseline = model_gscv.predict(x_lag)
                    x_cur_baseline = np.clip(x_cur_baseline, a_min=a_min, a_max=a_max)
                    x_cur = x_cur_baseline + white_noise
                    x_cur = np.clip(x_cur[0], a_min=0, a_max=None)

                    x_forecast.append(x_cur)

                x_forecast = np.array(x_forecast)
                x_forecasts.append(x_forecast)

                c_iter += 1
                progress_bar(c_iter, n_iters)

            x_forecasts = np.array(x_forecasts)

            x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
            x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
            x_forecast_pi_higher = np.quantile(x_forecasts, q=1 - alpha / 2, axis=0)

            rmse_forecast = root_mean_squared_error(x_test[:, idx_fp], x_forecast_median[:, idx_fp])
            rmse_forecasts.append(rmse_forecast)

            message = f"RMSE: {rmse_forecast:.2f}\n"
            message += f"GLC: {initial_glucose:.0f}, GLU: {initial_glutamine:.1f}\n"
            message += f"VCD: {initial_VCD:.1e}"

            # Plot the test values, forecast median, and prediction interval
            ax.scatter(t_test, x_test[:, idx_fp], label="Test", s=8, color="tab:orange", edgecolors="black", zorder=3)
            ax.plot(t_test, x_forecast_median[:, idx_fp], label=f"NVAR({p})", color="black", zorder=4)
            ax.fill_between(
                t_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_higher[:, idx_fp],
                label=f"{prediction_interval:.0%} PI", color="tab:blue", edgecolors="none", alpha=0.3, zorder=2
            )

            # Display forecast error with smaller font size and a clean background box to prevent clashing
            ax.text(
                x=0.03, y=0.95, s=message, va='top', ha='left', transform=ax.transAxes,
                fontsize=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7), zorder=5
            )

            y_max = max(y_max, np.max(x_test[:, idx_fp]), np.max(x_forecast_pi_higher[:, idx_fp]))

        # Apply axis limits and formatting to subplots
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                ax = axes[i, j]

                if not ax.has_data():
                    ax.set_visible(False)
                    continue

                ax.set_title(f"GLC={ic_glucose[i]:.1f}, GLU={ic_glutamine[j]:.1f}", fontsize=7)

                if i == axes.shape[0] - 1:
                    ax.set_xlabel("Time [h]", fontsize=8)
                    ax.xaxis.set_major_locator(MultipleLocator(4))

                if j == 0:
                    ax.set_ylabel(feature_plot, fontsize=8)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=7)


        fig.savefig(os.path.join("figures", f"figure_01_{model_type}_p{p:02d}_N{n_simulations:06d}.png"))
        plt.close(fig)

        # Calculate and report the forecast error
        rmse_mean = np.mean(rmse_forecasts)
        rmse_sd = np.std(rmse_forecasts)

        record = (model_type, p, n_simulations, rmse_mean, rmse_sd)
        records.append(record)


    cols_export = ["Model", "p", "N simulations", f"RMSE (mean) [{units}]", f"RMSE (SD) [{units}]"]
    df_export = pd.DataFrame.from_records(records, columns=cols_export)
    df_export.to_csv(os.path.join("tables", "table_01.csv"), index=False)

#Optimization
print("\nRunning final optimization using best model config...")
final_model_type = "knn"
final_p = 1
x_lag_train, x_cur_train, _, b_cur_train = generate_x_lag_cur_var(p=final_p, batch_ids=batch_ids, df=df_all,
                                                                  features=features, col_batch="Batch ID")
a_min = np.min(x_cur_train, axis=0)
a_max = np.max(x_cur_train, axis=0)

pipeline_final = make_pipeline(StandardScalerVAR(p=final_p), KNeighborsRegressor(n_neighbors=3, weights="uniform"))
pipeline_final.fit(x_lag_train, x_cur_train)
x_cur_train_pred = pipeline_final.predict(x_lag_train)
x_train_res = x_cur_train - x_cur_train_pred

bounds = [
    (ic_glucose.min(), ic_glucose.max()),
    (ic_glutamine.min(), ic_glutamine.max()),
    (ic_VCD.min(), ic_VCD.max()),
    (0.0, 400)
]


def objective_4var(vars_4):
    ic_glucose, ic_glutamine, ic_VCD, harvest_hours = vars_4
    return simulate(
        [ic_glucose, ic_glutamine, ic_VCD],
        pipeline_final, harvest_hours, x_train_res, a_min, a_max, final_p, features
    )


x0 = [np.median(ic_glucose), np.median(ic_glutamine), np.median(ic_VCD), 200]

result = minimize(objective_4var, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50})

opt_glc, opt_glu, opt_vcd, opt_harvest_hours = result.x
n_pi_sims = 100
pi_peaks = []
x_initial_opt = np.array([opt_glc, opt_glu, 0.0, 0.0, opt_vcd, opt_vcd, 0.0])
n_steps_opt = int(opt_harvest_hours / 2)

for _ in range(n_pi_sims):
    x_forecast = [x_initial_opt]
    while len(x_forecast) <= n_steps_opt:
        x_lag = x_forecast[::-1][:final_p].copy()
        while len(x_lag) < final_p:
            x_lag.append(x_lag[-1])
        x_lag = np.concatenate(x_lag, axis=0).reshape(1, -1)

        white_noise = np.array([rng.choice(x_train_res[:, j]) for j in range(x_train_res.shape[1])])
        x_cur_baseline = pipeline_final.predict(x_lag)
        x_cur_baseline = np.clip(x_cur_baseline, a_min=a_min, a_max=a_max)
        x_cur = np.clip(x_cur_baseline[0] + white_noise, a_min=0.0, a_max=None)
        x_forecast.append(x_cur)

    x_forecast = np.array(x_forecast)
    pi_peaks.append(np.max(x_forecast[:, idx_fp]))

pi_peaks = np.array(pi_peaks)
harvest_days = opt_harvest_hours / 24.0
downtime_days = 1.0
opt_stp, opt_sty, opt_profit, opt_cost, opt_val = calculate_economics(opt_glc, opt_glu, opt_vcd, opt_harvest_hours, np.median(pi_peaks))
val_dist = 1000 * pi_peaks
profit_dist = val_dist - opt_cost
stp_dist = profit_dist / (harvest_days + downtime_days)
sty_dist = pi_peaks / (harvest_days + downtime_days)

print("\n====Final Optimization Results & 95% Prediction Intervals====")
print(f"Recommended Initial Glucose: {opt_glc:.2f} mM")
print(f"Recommended Initial Glutamine: {opt_glu:.2f} mM")
print(f"Recommended Initial VCD: {opt_vcd:.2e} cells/mL")
print(f"Recommended Harvest Time: {opt_harvest_hours:.1f} hours ({harvest_days:.2f} days)")
print(f"Estimated Cell Culture Cost: ${opt_cost:.2f}/L")
print(
    f"Estimated Cell Culture Value at Harvest: ${opt_val:.2f}/L (95% PI: [${np.quantile(val_dist, 0.025):.2f}, ${np.quantile(val_dist, 0.975):.2f}])")
print(
    f"Estimated Space-Time Profit: ${opt_stp:.2f} / (L · d) (95% PI: [${np.quantile(stp_dist, 0.025):.2f}, ${np.quantile(stp_dist, 0.975):.2f}])")
print(
    f"Estimated Space-Time Yield: {opt_sty:.2f} mg / (L · d) (95% PI: [{np.quantile(sty_dist, 0.025):.2f}, {np.quantile(sty_dist, 0.975):.2f}])")