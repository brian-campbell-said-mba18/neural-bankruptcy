import os
import shutil
import time
import yaml
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from random_guess import RandomGuessAlgorithm


def load_yaml_and_save(yaml_path, run_path):
    with open(yaml_path, 'r') as f:
        config = yaml.load(f)
    save_path = os.path.join(run_path, 'cfg.yml')
    shutil.copyfile(yaml_path, save_path)
    return config


def load_dataset(year, shuffle=False):
    """Loads chosen data set, mixes it and returns."""
    main_path = './data/Dane/'
    file_name = '{}year.csv'.format(year)
    file_path = os.path.join(main_path, file_name)
    df = pd.read_csv(file_path, na_values='?')
    Y = df['class'].values
    X = df.drop('class', axis=1).values
    if shuffle:
        shuffled_idx = np.random.permutation(len(Y))
        X = X[shuffled_idx, :]
        Y = Y[shuffled_idx]
    return X, Y


def show_results(results, print_results=[], plot_roc=False):
    if (len(print_results) > 0):
        print('Results:')
        for metric in print_results:
            if type(results[metric]) == dict:
                print('{}={:.2f}, std={:,f}'.format(metric, results[metric]['mean'], results[metric]['std']))
            elif type(results[metric]) == str:
                print(results[metric])
            else:
                print('{}={:.2f}'.format(metric, results[metric]))

    if plot_roc:
        plt.plot(results['roc_curve']['fpr'], results['roc_curve']['tpr'])
        plt.show()


def do_experiment_for_one_year(run_path, year, config):
    """Performs the specified experiments for one year."""
    X, Y = load_dataset(year, shuffle=config['experiment']['shuffle_data'])
    if config['experiment']['type'] == 'single':
        X_train, Y_train, X_test, Y_test = split_into_train_test(X, Y, config['experiment']['test_share'])
        results = perform_one_experiment(X_train, Y_train, X_test, Y_test, config)
    elif config['experiment']['type'] == 'cv':
        results = perform_cv_runs(X, Y, config)

    results_path = os.path.join(run_path, 'results_year{}.pkl'.format(year))
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    show_results(results, **config['analysis'])


def perform_one_experiment(X_train, Y_train, X_test, Y_test, config):
    """Performs one experiment with a given data set and generates results."""
    algorithm_name = config['experiment']['algorithm']
    if algorithm_name == 'random_guess':
        algorithm = RandomGuessAlgorithm(**config['algo_params'])
    else:
        raise NotImplementedError('Algorithm {} is not an available option'.format(algorithm_name))

    # Perform experiment
    results = dict()
    results['fit_info'] = algorithm.fit(X_train, Y_train)
    pred_proba = algorithm.predict_proba(X_test)
    pred = np.argmax(pred_proba, axis=1)

    # Calculate and save results
    results['log_loss'] = metrics.log_loss(Y_test, pred_proba[:, 1])
    results['accuracy'] = metrics.accuracy_score(Y_test, pred)
    results['recall'] = metrics.recall_score(Y_test, pred)
    results['precision'] = metrics.precision_score(Y_test, pred)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, pred_proba[:, 1])
    results['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    results['roc_auc'] = metrics.auc(fpr, tpr)
    results['classification_report'] = metrics.classification_report(Y_test, pred)

    return results


def split_into_train_test(X, Y, test_share):
    split_point = int(len(Y) * test_share)
    X_test = X[:split_point, :]
    X_train = X[split_point:, :]
    Y_test = Y[:split_point]
    Y_train = Y[split_point:]
    return X_train, Y_train, X_test, Y_test


def perform_cv_runs(X, Y, config):
    # take data set
    # split up in cv and perform runs
    # concatenate results
    # returns averages
    raise NotImplementedError


def main(yaml_path='./config.yml', run_name=None):

    # Create output directory where experiment is saved
    if run_name is None:
        run_name = time.strftime('%Y%m%d-%H%M', time.localtime())
    run_path = os.path.join('./output', run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    config = load_yaml_and_save(yaml_path, run_path)

    # Do the specified experiments
    np.random.seed(config['experiment']['np_random_seed'])
    for year in config['experiment']['years']:
        do_experiment_for_one_year(run_path, year, config)


if __name__ == '__main__':
    main()
