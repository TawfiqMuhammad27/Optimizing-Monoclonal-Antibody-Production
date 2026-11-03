# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================


import sys

import numpy as np


def progress_bar(c_iter, n_iters, n_updates=1000, length_bar=20):
    update_every = max(1, round(n_iters / n_updates))
    if c_iter % update_every == 0:
        progress = c_iter / n_iters
        length_fill = round(length_bar * progress)
        length_void = length_bar - length_fill
        sys.stdout.write(f"\r{progress:6.0%} [" + length_fill * "=" + length_void * " " + "]")

def generate_x_lag_cur_var(p,batch_ids, df, features, col_batch="Batch ID",col_time="Time [h]"):
    x_lag=[]
    x_cur=[]
    t_cur=[]
    b_cur=[]

    for batch_id in batch_ids:
        mask=df.loc[:,col_batch]==batch_id
        x_batch=df.loc[mask,features].to_numpy()
        t_batch=df.loc[mask,col_time].to_numpy()
        b_batch=df.loc[mask,col_batch].to_numpy()

        current_values=x_batch[p:]
        current_times=t_batch[p:]
        current_batches=b_batch[p:]


        lagged_values=[]
        for i in range(1,p+1):
            lagged_values.append(x_batch[p-i:-i])

        lagged_values=np.concatenate(lagged_values,axis=1)

        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)
    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur


def calculate_economics(initial_glucose, initial_glutamine, initial_vcd, harvest_time_hours, predicted_mAb):
    harvest_time_days = harvest_time_hours / 24.0
    downtime_days = 1.0
    cell_culture_cost = (10 * initial_glucose) + (20 * initial_glutamine) + (0.0005 * initial_vcd)
    cell_culture_value = 1000 * predicted_mAb
    profit = cell_culture_value - cell_culture_cost
    space_time_profit = profit / (harvest_time_days + downtime_days)
    space_time_yield = predicted_mAb / (harvest_time_days + downtime_days)

    return space_time_profit, space_time_yield, profit, cell_culture_cost, cell_culture_value
def simulate(inputs,pipeline,time,x_train_res,a_min,a_max,p,features):
    ic_glucose,ic_glutamate,ic_VCD=inputs
    x_initial=np.array([ic_glucose,ic_glutamate,0.0,0.0,ic_VCD,ic_VCD,0.0])
    sims=10
    rng=np.random.default_rng(0)
    peaks=[]
    for i in range(sims):
        x_forecast=[x_initial]
        n_steps=int(time/2)

        for step in range(1,n_steps+1):
            x_lag=x_forecast[::-1][:p].copy()
            while len(x_lag)<p:
                x_lag.append(x_lag[-1])
            x_lag=np.concatenate(x_lag,axis=0).reshape(1,-1)
            white_noise = np.array([rng.choice(x_train_res[:, j]) for j in range(x_train_res.shape[1])])
            x_cur_baseline = pipeline.predict(x_lag)
            x_cur_baseline = np.clip(x_cur_baseline, a_min=a_min, a_max=a_max)
            x_cur = np.clip(x_cur_baseline[0] + white_noise, a_min=0.0, a_max=None)
            x_forecast.append(x_cur)
        x_forecast = np.array(x_forecast)
        mAb_col_idx = np.where(features == "mAb [mg/L]")[0][0]
        peaks.append(np.max(x_forecast[:, mAb_col_idx]))

    expected_mAb = np.mean(peaks)
    stp, sty, profit, cost, val = calculate_economics(
        ic_glucose, ic_glutamate, ic_VCD, time, expected_mAb
    )
    return -stp
