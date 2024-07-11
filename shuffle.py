from tensorflow.python.client import device_lib
from build_model import build_model, opt_search
import platform
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from collections import Counter
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from build_data import shape_data
from tqdm import tqdm
import copy
import gc


def compute_feature_importance(model, x_val, y_val, cols, fold):
    results = []
    print(' Computing LSTM feature importance...')

    # COMPUTE BASELINE (NO SHUFFLE)
    oof_preds = model.predict(x_val, verbose=0).squeeze()
    baseline_mae = np.mean(np.abs(oof_preds - y_val))
    results.append({'feature': 'BASELINE', 'mae': baseline_mae})

    for k in tqdm(range(len(cols))):
        gc.collect()
        # SHUFFLE FEATURE K
        save_col = x_val[:, :, k].copy()
        np.random.shuffle(x_val[:, :, k])

        # COMPUTE OOF MAE WITH FEATURE K SHUFFLED
        oof_preds = model.predict(x_val, verbose=0).squeeze()
        mae = np.mean(np.abs(oof_preds - y_val))
        results.append({'feature': cols[k], 'mae': mae})
        x_val[:, :, k] = save_col

    # DISPLAY LSTM FEATURE IMPORTANCE
    print()
    l = len(cols) + 1
    df = pd.DataFrame(results)
    df = df.sort_values('mae')
    plt.figure(4, figsize=(10, 20))
    plt.barh(np.arange(l), df.mae)
    plt.yticks(np.arange(l), df.feature.values)
    plt.title('LSTM Feature Importance', size=16)
    plt.ylim((-1, l))
    plt.plot([baseline_mae, baseline_mae], [-1, l], '--', color='orange',
             label=f'Baseline OOF\nMAE={baseline_mae:.3f}')
    plt.xlabel(f'Fold {fold + 1} OOF MAE with feature permuted', size=14)
    plt.ylabel('Feature', size=14)
    plt.legend()
    plt.show()


def plothistories(histories, regression):
    for history in histories:
        # summarize history for accuracy and loss
        plt.figure(3)
        plt.subplot(2, 1, 1)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.subplot(2, 1, 2)
        if regression:
            plt.plot(history['root_mean_squared_error'])
            plt.plot(history['val_root_mean_squared_error'])
            plt.title('model error')
            plt.ylabel('rmse')
        else:
            plt.plot(history['accuracy'])
            plt.plot(history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def plotauc(y_pred_p, yval_p, mess):

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(yval_p, y_pred_p)
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fpr_keras, tpr_keras, label='RF (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve {mess}')
    plt.legend(loc='best')
    plt.show()
    y_inverted = copy.deepcopy(y_pred_p)
    y_val_inv = copy.deepcopy(yval_p)
    for idx, y in enumerate(y_pred_p):
        y_inverted[idx] = 1-y
        y_val_inv[idx] = 1-y_val_inv[idx]
    fnr_keras, tnr_keras, thresholds_keras = roc_curve(y_val_inv, y_inverted)

    auc_keras = auc(fnr_keras, tnr_keras)

    plt.figure(2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fnr_keras, tnr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.plot(fnr_keras, tnr_keras, label='RF (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False negative rate')
    plt.ylabel('True negative rate')
    plt.title(f'ROC curve {mess}')
    plt.legend(loc='best')
    plt.show()


def shuffle_and_train(x_adj, y_adj, params):
    timesteps = params[2]
    regression = params[4]
    reshuffle = params[6]
    use_checkpoints = params[7]
    train_single = params[8]
    features = params[3]
    tag = params[16]
    labels_wnd = params[17]
    restore_hist = params[18]
    feature_importance = params[19]
    model_opt_search = params[20]
    trials = 20
    # save some data for testing
    # train_idx = int(cut * split)
    if reshuffle:
        x_adj = shape_data(x_adj, params=params, training=True)
        # balance 50/50 and shuffle pos and neg examples
        np.random.seed(42)
        shuffle_index = np.random.permutation(x_adj.shape[0])
        x_adj, y_adj = x_adj[shuffle_index], y_adj[shuffle_index]

        # find indexes of each label
        idx_1 = np.argwhere(y_adj > 0).flatten()
        idx_0 = np.argwhere(y_adj <= 0).flatten()

        shuffle_1 = np.random.permutation(len(idx_1))
        shuffle_0 = np.random.permutation(len(idx_0))
        minlen = min(len(idx_1), len(idx_0))
        if len(idx_1) > len(idx_0):
            idx_1 = idx_1[shuffle_0]
        else:
            idx_0 = idx_0[shuffle_1]
        # shuffle_index = np.random.permutation(minlen)

        # grab specified cut of each label put them together
        x_adj = np.concatenate((x_adj[idx_1[:minlen]], x_adj[idx_0[:minlen]]), axis=0)
        # X_test = np.concatenate((x_adj[idx_1[train_idx:cut]], x_adj[idx_0[train_idx:cut]]), axis=0)
        y_adj = np.concatenate((y_adj[idx_1[:minlen]], y_adj[idx_0[:minlen]]), axis=0)
        # y_test = np.concatenate((y_adj[idx_1[train_idx:cut]], y_adj[idx_0[train_idx:cut]]), axis=0)

        # shuffle again to mix labels
        np.random.seed(42)
        if regression:
            cp = Counter(num > 0 for num in y_adj)
            # labels_n = []
            print(f'balanced positive examples % {100*cp[True]/(cp[True]+cp[False]):.2f}')
        else:
            cp = Counter(y_adj)
            print(f'balanced examples {cp.most_common(2)}')
        shuffle_index = np.random.permutation(x_adj.shape[0])
        x_adj, y_adj = x_adj[shuffle_index], y_adj[shuffle_index]

    gpus = tf.config.list_physical_devices('GPU')
    # cpus = tf.config.list_physical_devices('CPU')

    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]

    print(f'available devices: {get_available_devices()}')
    if gpus:
        if platform.system() == "Windows":
            print(f'will compute on {tf.test.gpu_device_name()}')
            print(f'eager mode is { tf.executing_eagerly()}')
            #   tf.config.set_logical_device_configuration(
            #    gpus[0],
            #    [tf.config.LogicalDeviceConfiguration(memory_limit=2800)])

    # tf.debugging.set_log_device_placement(True)
    # strategy = tf.distribute.MirroredStrategy(cpus)
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    histories = list()
    # with strategy.scope():

    for index, (train_indices, val_indices) in enumerate(skf.split(x_adj, y_adj)):
        # following code is for kfold
        if index > 0:
            continue
        print("Training on fold " + str(index + 1) + "/5...")
        # Generate batches from indices
        xtrain, xval = x_adj[train_indices], x_adj[val_indices]
        ytrain, yval = y_adj[train_indices], y_adj[val_indices]
        if reshuffle:
            pass
            # bins = np.linspace(min(y_adj), max(y_adj), 5)
            # y_binned = np.digitize(y_adj, bins)

            #xtrain, xval, ytrain, yval = train_test_split(x_adj, y_adj, test_size=0.2, shuffle=True)


            # ytrain, yval = to_categorical(ytrain, 2), to_categorical(yval, 2)
            # graph(xtrain[0], hold_g=ytrain.T[0], start_g=timesteps, len_g=50, col_type=0)
            #np.save(f'xtrain_{tag}', xtrain, allow_pickle=True)
            #np.save(f'ytrain_{tag}', ytrain, allow_pickle=True)
            #np.save(f'xval_{tag}', xval, allow_pickle=True)
            #np.save(f'yval_{tag}', yval, allow_pickle=True)

        #xtrain, xval = np.load(f'xtrain_{tag}.npy', allow_pickle=True), \
        #    np.load(f'xval_{tag}.npy', allow_pickle=True)
        #ytrain, yval = np.load(f'ytrain_{tag}.npy', allow_pickle=True), \
        #    np.load(f'yval_{tag}.npy', allow_pickle=True)

        if np.count_nonzero(np.isnan(xtrain)) > 0:
            raise Exception('Nans found in data')

        if not regression:
            ytrain, yval = to_categorical(ytrain, 2), to_categorical(yval, 2)
        # checkpoint_filepath = f'timeseries_bayes_opt_POC/trial_05/checkpoint'
        checkpoint_filepath = f'checkpoint_{tag}_l{labels_wnd}.weights.h5'
        if regression:
            monitor = 'val_loss'
            mode = 'min'
        else:
            monitor = 'val_loss'
            mode = 'min'
        for lr in range(1, 2):

            model_checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor=monitor,
                mode=mode,
                save_best_only=True)
            early_stop = EarlyStopping(monitor='val_loss',
                                       patience=4,
                                       verbose=1,
                                       min_delta=1e-4,
                                       restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.2,
                                          patience=3,
                                          verbose=1,
                                          min_delta=1e-3,
                                          mode='min')
        if train_single:
            model = build_model(xtrain, regression=regression, learning_rate=0.003, features=xtrain.shape[2])
            if os.path.isfile(checkpoint_filepath) and use_checkpoints:
                model.load_weights(checkpoint_filepath)
                print(f'loaded weights')
            #device = cuda.get_current_device()
            #cuda.select_device(0)
            #device.reset()
            gc.collect()
            # with tf.device('/gpu:0'):
            history = model.fit(xtrain, ytrain,
                                epochs=300,
                                batch_size=64,
                                shuffle=True,
                                validation_data=(xval, yval),
                                callbacks=[model_checkpoint_callback,
                                           early_stop,
                                           reduce_lr])
            model.load_weights(checkpoint_filepath)
            if history.history['val_loss'][-1] > 0.68:
                raise Exception('Model stopped too early')


            #y_pred_p = model.predict(xval)
            #plotauc(y_pred_p.T[0], yval.T[0], 'val data')
            histories.append(history)
            if not os.path.exists('train_history'):
                os.mkdir('train_history')
            pd.DataFrame(history.history).to_pickle(f"train_history/train_history{tag}_l{labels_wnd}_{index}.pkl")
            model.save(f'models/my_model_{tag}_l{labels_wnd}_{index}.keras')
            #plothistories([history.history], regression)
        if feature_importance:
            cols = ['adx_ps', ' adx_ns', ' adx_pm', ' adx_nm', ' adx_pl', ' adx_nl',
                    'boll_ps', ' boll_pm', ' boll_pl',
                    'dcp_s', ' dcp_m', ' dcp_l',
                    'stoch_rsi_s', ' stoch_rsi_m', ' stoch_rsi_l',
                    'stoch_s', ' stoch_m', ' stoch_l',
                    ' will_s', ' will_m', ' will_l',
                    'ktc',
                    'tsi']
            feat_nums = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14]
            bad_features = [23, 24, 26, 27, 28]
            fcols = list()
            for i, c in enumerate(cols):
                if i in feat_nums:
                    fcols.append(c)

            # fcols = [*range(30)]
            lstm = load_model(f'models/my_model_{tag}_l{labels_wnd}_{index}.keras')
            compute_feature_importance(lstm, xval, yval, cols=fcols, fold=0)

        if restore_hist:
            history = pd.read_pickle(f"train_history/train_history{tag}_l{labels_wnd}_{index}.pkl")
            plothistories([history], regression)

        if model_opt_search:
            opt_search(xtrain, ytrain, xval, yval, trials)

