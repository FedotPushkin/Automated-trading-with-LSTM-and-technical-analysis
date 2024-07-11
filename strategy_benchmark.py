import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
# import plotly.graph_objs as go
import copy
import math


def avg_hold(hold):
    # y_pred contains 1 for long, -1 for short, 0 for wait, this method returns average time for all states
    wa = 0
    temp_p = [0]
    temp_n = [0]
    for i, h in enumerate(hold):
        if i > 0:
            if h > 0:
                temp_p[-1] += 1
                if hold[i - 1] != h:
                    temp_p.append(0)
            elif h < 0:
                temp_n[-1] += 1
                if hold[i - 1] != h:
                    temp_n.append(0)

        if h == 0:
            wa += 1
    return temp_p, temp_n, wa


def strategy_bench(preds, start_pos, verb=False, deltab=0, deltas=0, candles=None, hists=False):

    start_balance = 1000
    start_short_balance = 1000
    balance = start_balance
    profit_long = []
    leftover = 0
    short_leftover = 0
    profit_short = []
    amount = 0
    amount_short = 0
    trades = 0
    trades_short = 0
    fee = 0.00/100
    # 'long':1,'short':-1,'wait:0
    position = 0
    holding = list()
    for i in range(start_pos, start_pos+preds.size-1):

        if preds[i - start_pos] is None:
            raise Exception('some prediction is none')
        if position == 0:
            #   #open long#
            if preds[i-start_pos] > 0.5+deltab:

                #   amount available after opening long position
                amount = start_balance/(candles.iloc[i]['open']*(1+fee))
                balance = 0
                position = 1
                holding.append(1)
                if verb:
                    print('long at '+str(candles.iloc[i]['Date'])+' for ' + str(candles.iloc[i]['open']))
            #   #open short#
            if preds[i - start_pos] < 0.5 - deltas:
                position = -1
                # how much was sold
                amount_short = start_short_balance / (candles.iloc[i]['open'] * (1 - fee))
                holding.append(-1)
                if verb:
                    print('short at ' + str(candles.iloc[i]['Date']) + ' for ' + str(candles.iloc[i]['open']))

        else:
            # #close long#
            if position == 1 and preds[i-start_pos] < 0.5+deltas:
                abs_prof = (1-fee)*amount*candles.iloc[i]['open']-start_balance
                profit_long.append(abs_prof/start_balance)
                leftover += abs_prof
                trades += 1
                balance = start_balance
                amount = 0
                position = 0
                holding.append(0)
                if verb:
                    print('short at ' + str(candles.iloc[i]['Date']) + ' for ' + str(candles.iloc[i]['open']))

            #   #close short#
            if position == - 1 and preds[i-start_pos] > 0.5-deltab:
                position = 0
                curr_profit = start_short_balance-(1 + fee) * amount_short * candles.iloc[i]['open']
                short_leftover += curr_profit
                profit_short.append(curr_profit/start_short_balance)
                amount_short = 0
                holding.append(0)
                trades_short += 1
                if verb:
                    print('long at ' + str(candles.iloc[i]['Date']) + ' for ' + str(candles.iloc[i]['open']))
            else:
                holding.append(position)

    if 1:
        print('balance ', balance, ' amount ', amount, ' trades ', trades,
              ' amount short ', amount_short, ' trades_short ', trades_short, ' deltab', deltab, ' deltas', deltas)
    result = ((balance+leftover+amount*(1-fee)*candles['close'][candles.shape[0]-1])/start_balance)-1
    if amount_short > 0:
        short_leftover += start_short_balance - amount_short * (1+fee)*candles['close'][candles.shape[0]-1]
    result_short = short_leftover/start_short_balance
    if verb:
        if amount > 0:
            print('liqudating long at  ' + str(candles['close'][candles.shape[0]-1]))
        if amount_short > 0:
            print('liqudating short at  ' + str(candles['close'][candles.shape[0]-1]))

    l, s, wait = avg_hold(holding)
    print(f' avg profit long {np.mean(profit_long)}, avg len {np.mean(l):.2f}')
    print(f' avg profit short {np.mean(profit_short)}, avg len {np.mean(s):.2f}')
    print(f'wait {100*wait/len(holding):.2f} %')

    prof = 0
    for p in profit_long:
        if p > 0:
            prof += 1
    if len(profit_long) > 0:
        print('winrate long ' + str(prof/len(profit_long)))
    prof = 0
    for p in profit_short:
        if p > 0:
            prof += 1
    if len(profit_short) > 0:
        print('winrate short ' + str(prof / len(profit_short)))
    if hists:
        trace1 = px.histogram(profit_long, nbins=400, title='longs')
        trace2 = px.histogram(profit_short, nbins=400, title='shorts')
        trace3 = px.histogram(l, nbins=10, title='longs')
        trace4 = px.histogram(s, nbins=10, title='shorts')
        trace1.show()
        trace2.show()
        trace3.show()
        trace4.show()

    def sign(a):
        if a > 0:
            return '+'
        else:
            return''
    result_str = f'long {sign(result)} {result*100:.2f}%, short {sign(result_short)}{result_short*100:.2f}%'
    print(result_str)
    return result, np.mean(profit_long), result_short, np.mean(profit_short), holding


def random_guess(length, start, preds):
    # launches strategybench many times on random long/short orders and shows average result
    res = []
    res_s = []

    for _ in range(10):
        avg_hold_dur = 8
        y_rand = list()
        while len(y_rand) < length:
            n = np.random.random_integers(low=0, high=1)
            hold = np.random.random_integers(low=4, high=10)
            for k in range(hold):
                y_rand.append(n)
        # rand_preds = np.random.random_integers(low=0, high=1, size=length)
        a, b, _ = strategy_bench(preds=np.array(y_rand), start_pos=start, verb=False)
        res.append(a)
        res_s.append(b)
    print('mean long', np.mean(res), 'mean short', np.mean(res_s))


def spoil_labels(labels, rate):
    spoiled = copy.deepcopy(labels)
    for i in range(labels.size):
        r = np.random.random_integers(low=1, high=100)
        if r < rate:
            spoiled[i] = (labels[i]+1) % 2
    return spoiled


def improve_labels(labels, true_labels, rate):
    improved_l = copy.deepcopy(labels)
    improve_count = math.ceil(rate*labels.size/100)
    i = 0
    points = set()
    while i < improve_count:
        point = np.random.random_integers(low=0, high=labels.size-1)
        if abs(labels[point]-true_labels[point]) > 0.5 and point not in points:
            improved_l[point] = true_labels[point]
            points.add(point)
            i += 1
    return improved_l


def bench_cand(pred, timesteps):
    candles = np.load('candles.npy', allow_pickle=True)
    start_pos = len(candles)-timesteps-100

    profit = list()
    # for j in range(0, 101):
    temp_profit = 0
    for s in range(start_pos, start_pos + len(pred) - 1):
        if pred[s-start_pos] is None:
            pass
        elif pred[s-start_pos]:
            temp_profit += ((1-0.001)*candles[1][s]-(1+0.001)*candles[0][s])/candles[0][s]
    profit.append(temp_profit)
    plt.figure(1)
    plt.plot(profit)
    plt.title('profit ')
    plt.ylabel('sum roi')
    plt.xlabel('pred value(0.3-0.9')
    plt.show()
