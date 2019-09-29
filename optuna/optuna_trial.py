import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
from sklearn import datasets

import matplotlib.pyplot as plt

import sqlite3

args = {
        "train_test_ratio" : 0.8,
        "initial_learning_rate" : 0.01,
        "batch_size" : 64,
        "epochs" : 10,
        "decay_steps" : 200,
        "decay_rate" : 0.9
}

def load_data(train_rate):
    dataset = datasets.load_diabetes()
    X = dataset['data']
    y = dataset['target']

    X_sc = X - X.mean(axis=0)
    X_sc = X_sc / X.std(axis=0)

    y_sc = y -y.mean()
    y_sc = y_sc / y.std()

    n_trains = int(X.shape[0] * train_rate)
    X_train = X_sc[:n_trains]
    y_train = y_sc[:n_trains]
    X_test = X_sc[n_trains:]
    y_test = y_sc[n_trains:]

    return X_train, y_train, X_test, y_test

def create_model(trial, n_inputs, n_outputs):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)
    n_units = trial.suggest_int("n_units",  3, 5)

    layers = []
    layers.append(
        tf.keras.layers.Dense(n_inputs, activation="relu")
    )
    for i in range(n_layers):
        layers.append(
            tf.keras.layers.Dense(2 ** n_units, activation="relu")
        )
        layers.append(
            tf.keras.layers.Dropout(rate=dropout_rate)
        )

    layers.append(
        tf.keras.layers.Dense(n_outputs, activation=None)
    )

    return tf.keras.Sequential(layers)

def create_optimizer(trial):
    initial_lr = args["initial_learning_rate"]
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=args["decay_steps"],
        decay_rate=args["decay_rate"],
        staircase=True
    )
    
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    if optimizer_name == "Adam":
      optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif optimizer_name == "RMSprop":
      optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    
    return optimizer

def trainer(trial, X_train, y_train, n_inputs, n_outputs):
  
    model = create_model(trial, n_inputs, n_outputs)
    optimizer = create_optimizer(trial)

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        #metrics=["mse"]
    )

    model.fit(
        x=X_train,
        y=y_train,
        batch_size=args["batch_size"],
        epochs=args["epochs"]
    )

    model.fit = tf.function(model.fit)
    
    return model

def objective(trial):
  
    X_train, y_train, X_test, y_test = load_data(args["train_test_ratio"])
    
    model = trainer(trial, X_train, y_train, X_train.shape[1], 1)
    
    evaluate = model.evaluate(x=X_test, y=y_test, batch_size=args["batch_size"])
  
    return evaluate

def main():

    db_name = "diabetes_trial.db"
    conn = sqlite3.connect(db_name)

    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner()

    study = optuna.create_study(
        study_name="diabetes_trial",
        storage="sqlite:///{}".format(db_name),
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )
    study.optimize(objective, n_trials=50)

    conn.close()

if __name__ == "__main__":
    main()