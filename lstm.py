import numpy as np
import pandas as pd
from collections import Counter
from generate_labels import Genlabels
import matplotlib.pyplot as plt
import copy
from build_data import build_tt_data
from shuffle import shuffle_and_train
from xgboost import XGBClassifier
import math
from ta.trend import sma_indicator
from graph import graph
from strategy_benchmark import strategy_bench, random_guess, improve_labels
from predict import getpreds, predict_val
from get_data import get_data_files
# from verstack.stratified_continuous_split import scsplit
# sys.path.append(os.path.join('historical_data'))
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


if __name__ == '__main__':
    start = '01 Jan 2017'
    end = '27 Jun 2024'
    download_candles = 0
    reload_labels = 1
    build_x = 0
    reshuffle = 1
    train = 1
    train_single = 1
    feature_importance = 0
    use_checkpoints = 0
    restore_hist = 0
    opt_search = 0
    predict = 1
    calc_xval = 0
    bench = 1
    regression = 0
    validation_length = 10000
    validation_lag = 31
    timesteps = 35
    labels_wnd = 7
    features = 23
    currency = 'BTC'
    granularity = 30
    tag = f't{timesteps}_f{features}_{currency}_g{granularity}'
    long, med, short = 26, 12, 9
    hl, ll, sl, ss = math.ceil(long / 2), math.ceil(long * 1.27), math.ceil(long * 0.77), math.ceil(short / 2)
    params = [validation_length, validation_lag, timesteps, features, regression, calc_xval, reshuffle, use_checkpoints,
              train_single, long, med, short, hl, ll, sl, ss, tag, labels_wnd, restore_hist, feature_importance, opt_search, currency, granularity]

    if download_candles:
        candles = get_data_files(start, end, granularity, currency)
    else:

        candles = pd.read_hdf(f'historical_data/candles_{currency}_g{granularity}.h5', 'candles')
    if reload_labels:
        labels_p, labels_n = Genlabels(candles['close'], window=labels_wnd, polyorder=3).labels
        if regression:
            cp = Counter(num > 0 for num in labels_p)

            print(f'initial positive examples % {100*cp[True]/(cp[True]+cp[False]):.2f}')
        else:
            cp = Counter(labels_p)
            # cn = Counter(labels_n)
            # print(f'initial pos {cp.most_common(2)}')
            # print(f'initial neg {cn.most_common(2)}')

        y_p = labels_p[validation_lag+timesteps:- validation_length]
        y_n = labels_n[validation_lag+timesteps:- validation_length]
        np.save('historical_data/y_p', y_p, allow_pickle=True)
        np.save('historical_data/y_n', y_n, allow_pickle=True)
    if build_x:
        X = build_tt_data(candles, params)
        np.save(f'historical_data/X_{tag}', X, allow_pickle=True)
    xp, y_p, y_n, = np.load(f'historical_data/X_{tag}.npy', allow_pickle=True), \
        np.load('historical_data/y_p.npy', allow_pickle=True), \
        np.load('historical_data/y_n.npy', allow_pickle=True), \

    if xp.shape[1] != timesteps:
        raise Exception('timesteps dont match xp')
    if xp.shape[2] != features:
        raise Exception('features dont match xp')

    if train:
        #   ensure equal number of labels, shuffle, and split
        shuffle_and_train(xp, y_p, params)
        # shuffle_and_train(xn, y_n, 'neg')
    candles = pd.DataFrame(candles)
    if predict:
        predict_val(params)

    y_pred_p = getpreds(tag)

    hitp, hitn, missp,  missn = 0, 0, 0, 0

    candles_val = candles[-validation_length:-timesteps]
    savgol = Genlabels(candles_val['close'], window=labels_wnd, polyorder=3).apply_filter(deriv=0, hist=candles_val['close'])
    savgol_deriv = Genlabels(candles_val['close'], window=labels_wnd, polyorder=3).apply_filter(deriv=1, hist=candles_val['close'])
    labels_p, labels_n = Genlabels(candles['close'], window=labels_wnd, polyorder=3).labels
    lag = labels_p.shape[0]-validation_length+validation_lag+timesteps
    balp = 0
    baln = 0
    preds_len = validation_length-validation_lag-timesteps-1

    for i in range(preds_len):  # validation_length - validation_lag - 2 * timesteps-1):

        delta = 0.02
        if y_pred_p[i] < 0.5-delta:
            baln += 1
            if labels_p[i+lag] == 0:
                hitn += 1
            else:
                missp += 1
        elif y_pred_p[i] > 0.5+delta:
            balp += 1
            if labels_p[i+lag] == 1:
                hitp += 1
            else:
                missn += 1

    # plotauc(y_pred_p[:preds_len], labels_p[lag:lag+preds_len], 'check data')
    precision_p = hitp / (hitp + missp)
    recall_p = hitp / (hitp + missn)
    precision_n = hitn / (hitn + missn)
    recall_n = hitn / (hitn + missp)
    f1_score_p = 2 * (precision_p * recall_p) / (precision_p + recall_p)
    f1_score_n = 2 * (precision_n * recall_n) / (precision_n + recall_n)
    print(f'Precision p  {precision_p:.2f}')
    print(f'Recall p  {recall_p:.2f}')
    print(f'f1 p {f1_score_p:.2f}')
    print(f'Precision n {precision_n:.2f}')
    print(f'Recall n {recall_n:.2f}')
    print(f'f1 n {f1_score_n:.2f}')
    print(f'preds bal {baln / (balp + baln):.2f} zeros')
    cp = Counter(labels_p[lag:lag+preds_len])
    # cn = Counter(labels_n)
    zeros = cp.most_common(2)[0][1]
    ones = cp.most_common(2)[1][1]
    print(f'distrib {cp.most_common(2)} zeros {zeros/(zeros+ones):.2f} ones {ones/(zeros+ones):.2f}')
    target = []

    start = candles.shape[0] - validation_length
    if bench:
        # start = candles.shape[0] - validation_length
        # random_guess(y_pred)
        # random_guess(labels_p[lag:].size, start+validation_lag+timesteps, labels_p[lag:])
        best = -10000000
        hold_best, holding_m = [], []
        best_params = [0, 0]
        profit_p, profit_n, avg_prof_l, avg_prof_s = list(), list(), list(), list()
        hold_l = np.ones(len(y_pred_p))
        #   random_guess(y_pred_p.shape[0], start+validation_lag+timesteps, None)

        # y_improved = improve_labels(y_pred_p, labels_p[lag:], 5)
        lower = 30
        upper = 200
        mul = 0.002
        # y_pred_p = invert(y_pred_p)
        for i in range(lower, upper):

            # y_improved = improve_labels(y_pred_p, labels_p[lag:], 5)
            result_l, avg_l, result_s, avg_s, holding_m = strategy_bench(preds=y_pred_p,
                                                                           # preds=labels_p[lag:],
                                                                           start_pos=start+validation_lag+timesteps,
                                                                           verb=False,
                                                                           deltab=i*mul,
                                                                           deltas=i*mul,
                                                                           candles=candles,
                                                                           hists=(i == (upper-lower)//2))
            profit_p.append(result_l*100)
            profit_n.append(result_s*100)
            avg_prof_l.append(avg_l)
            avg_prof_s.append(avg_s)
            if i == 165:
                test_hold = holding_m
        print(f'holding long')
        strategy_bench(preds=hold_l, start_pos=start + validation_lag + timesteps, candles=candles)
        labels = [f'{(i *mul):.2f}' for i in range(lower, upper, math.ceil(0.1*(upper-lower)))]
        # ticks = [(i * mul) for i in range(lower, upper)]
        ticks = [*range(0, upper-lower, math.ceil(0.1*(upper-lower)))]
        plt.subplot(2, 2, 1)
        plt.ylim(min(ticks), max(ticks))
        plt.xticks(ticks=ticks, labels=labels, rotation='vertical')
        plt.plot(profit_p)
        plt.plot(np.zeros(upper-lower))
        plt.ylim(-20, 100)
        plt.title('long')
        plt.ylabel('profit %', )
        plt.subplot(2, 2, 2)
        plt.xticks(ticks=ticks, labels=labels, rotation='vertical')
        plt.plot(profit_n)
        plt.plot(np.zeros(upper-lower))
        plt.ylim(-20, 100)
        plt.title('short')
        plt.subplot(2, 2, 3)
        plt.xticks(ticks=ticks, labels=labels, rotation='vertical')
        plt.plot(avg_prof_l)
        plt.plot(np.zeros(upper - lower))

        plt.ylabel('avg profit per trade', )
        plt.subplot(2, 2, 4)
        plt.xticks(ticks=ticks, labels=labels, rotation='vertical')
        plt.plot(avg_prof_s)
        plt.plot(np.zeros(upper - lower))


        plt.suptitle(currency)
        plt.show()
        print(f'best score at {best_params}, result {best}')
        np.save("holding", test_hold, allow_pickle=True)

    else:

        test_hold = np.load("holding.npy")

    lines = [savgol, savgol_deriv, y_pred_p, labels_p[lag:], test_hold]
    graph(lines=lines, start_g=start, lag=200, col_type=1, params=params)
    #graph(lines=lines, start_g=start, lag=200, col_type=1, params=params)
    #graph(lines=lines, start_g=start, lag=400, col_type=1, params=params)

    def plot_price_to_volume(price, volume, smooth_wnd):
        sma_vol = pd.Series(sma_indicator(volume, window=smooth_wnd, fillna=False))
        sma_cl = pd.Series(sma_indicator(price, window=smooth_wnd, fillna=False))
        plt.subplot(2, 1, 1)
        plt.plot(sma_vol)

        plt.ylabel('vol')
        plt.xlabel('time')

        plt.subplot(2, 1, 2)
        plt.plot(sma_cl)
        plt.ylabel('price')
        plt.xlabel('time')
        plt.show()


    def invert(y):
        y_inverted = copy.deepcopy(y)
        for idx, y in enumerate(y):
            y_inverted[idx] = 1 - y
        return y_inverted

