import numpy as np
import pandas as pd
import time
from keras.models import load_model
from build_data import build_val_data, shape_data
import joblib
from datetime import datetime


def predict_val(params):
    validation_length, validation_lag, timesteps, features, regression, calc_xval = params[:6]
    labels_win = params[17]
    tag = params[16]
    currency = params[21]
    granularity = params[22]
    candles = pd.read_hdf(f'historical_data/candles_{currency}_g{granularity}.h5', 'candles')
    # print(lstm.summary())
    # model.load_weights(f'timeseries_bayes_opt_POC/trial_00/checkpoint')
    runs = 0
    t_sum = 0
    change_features_n = False
    print(f'start val calc { datetime.now()}')
    if calc_xval:
        val_sample = validation_length
        x_v_temp = list()
        length = validation_length - validation_lag - timesteps

        for v in range(length):
            t0 = time.time()
            s = -validation_length + v
            e = -validation_length + v + validation_lag + timesteps
            val_input = candles.iloc[s:e]
            cl_plus = candles.iloc[s - 24:e]['close'].astype('float64')

            x_val_single = build_val_data(data=val_input, clplus=cl_plus, params=params)
            x_v_temp.append(x_val_single)

            t1 = time.time()
            runs += 1
            curr_lap = t1-t0
            t_sum += curr_lap
            avg_lap = t_sum/(v+1)
            if v % (length//100) == 0:
                print(f'adding val data {v} of {val_sample}, eta {curr_lap*(val_sample-v)/60:.2f} min, '
                      f'avg lap: {avg_lap * 100:.2f} ms, curr lap {curr_lap * 100:.2f} ms')

        x_val = np.array(x_v_temp)
        np.save(f'historical_data/x_val_{tag}_nonscaled.npy', x_val, allow_pickle=True)
        scaler = joblib.load(f'models/scaler_{tag}.dump')
        x_val = shape_data(x_val, params=params, training=False, scaler=scaler)
    else:
        x_val = np.load(f'historical_data/x_val_{tag}_nonscaled.npy', allow_pickle=True)

        scaler = joblib.load(f'models/scaler_{tag}.dump')
        x_val = shape_data(x_val, params=params, training=False, scaler=scaler)
        if change_features_n:
            x_val_n = np.empty(shape=(1, timesteps, features))
            for ind, x in enumerate(x_val):
                feat_nums = [range(19)]
                drop_features = [2, 5, 8, 10, 11]
                cut_timesteps = 5
                onestep = np.array([x[:, np.setdiff1d(feat_nums, drop_features)][:-cut_timesteps]])
                x_val_n = np.append(x_val_n, onestep, axis=0)
                print(f'cutting xval, {ind} of {x_val.shape[0]}')
            x_val_n = np.delete(x_val_n, 0, axis=0)
            x_val = x_val_n
    pred_skf = list()
    for i in range(0, 1):
        lstm = load_model(f'models/my_model_{tag}_l{labels_win}_{i}.keras')
        preds = lstm.predict(x_val, batch_size=64)
        pred_skf.append(preds .T[1])

    preds_p = np.array(pred_skf)[0].T
    np.save(f'preds_p_{tag}', preds_p, allow_pickle=True)
    # np.save(f'y_strat_{tag}_n', y_strat_n, allow_pickle=True)


def getpreds(tag):
    y_pred = np.load(f'preds_p_{tag}.npy', allow_pickle=True)

    #for col in range(y_strat.shape[0] - 1):
    #    column = y_strat[col]
        # for i in range(column.shape[0]-1):
        # if np.isnan(column[i]):
        # column =np.delete(column, i)
    #    vec_p = np.nanmean(column)
    #    y_pred.append(vec_p)  # statistics.mean(vec))

    return np.array(y_pred)


def preds2binary(pred):
    if np.isnan(pred):
        return
    for v in range(len(pred)):
        if pred[v] < 0.4:
            pass  # pred[v] = 1
        else:
            if pred[v] > 0.6:
                pass  # pred[v] = 0
            else:
                pass  # pred[v] = None

    return pred


def addnones(pred, params):
    val_lag, timesteps = params[1:3]
    for _ in range(timesteps + val_lag):
        pred = np.insert(pred, 0, axis=0)
    for _ in range(timesteps):
        pred = np.append(pred, np.nan)
    return pred


def makegrad(pred):
    grad = []
    for pr in range(1, pred.shape[0]):
        fragment = pred[pr - 1:pr + 1]
        frag_grad = np.gradient(fragment, [0, 1])
        grad.append(frag_grad[0])
    return grad
