
import argparse
from tqdm import trange
from pathlib import Path

import matplotlib.pyplot as plt

#import numpy as np
import cupy as np

from .network import WeirdNetwork
from .datasets import get_dataset, get_accuracy
from experiments.exp_subsets import make_models, pretraining_experiment


def train(model, epochs, acc_threshold, graph_it):
    #fetch data
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')

    cost_history = model.train(X_train, Y_train, epochs, acc_threshold)

    correct, total = get_accuracy(model, X_test, Y_test)
    print(f"average test error: {model.evaluate(X_test, Y_test)/total}")
    print(f"total test samples: {total}\nnumber correct: {correct}\naccuracy: {correct/total}")

    if graph_it:
        plt.title("training progression")
        plt.ylabel("cost")
        plt.xlabel("epoch")
        plt.plot(list(range(epochs)), cost_history)
        plt.show()

    return model

def build_model(fname):
    return WeirdNetwork.create_from_config(Path(fname))

def run(model, inp_fname):
    print("under construction: that file had better contain a matrix lol")
    with open(inp_fname, 'rb') as f:
        #TODO: use a different ingestion method based on the type of file
        #   pull it into a np vector!
        buffer = f.read()
    return model.predict(buffer)

def experiment(config, epochs):
    pretraining_experiment(epochs)
    

def play(config):
    make_models(100, config)




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Experiment with a WeirdNetwork.')
    parser.add_argument('action',type=str, choices={"train", "experiment", "run", "play"},
                        help='action to take')
    parser.add_argument('--load', type=str,
                        help='load a saved model file')
    parser.add_argument('--save', type=str,
                        help='save a model to a file')
    parser.add_argument('--config', type=str,
                        help='load a model or experiment configuration')
    # training only
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs to run')
    parser.add_argument('--graph', action="store_true",
                        help='graph the cost over training')
    parser.add_argument('--accuracy', type=int, default=100,
                        help='accuracy threshold at which to stop training')
    # run/play only
    parser.add_argument('--input', type=str,
                        help='model input')

    args = parser.parse_args()

    if args.action == "train":
        model = None
        if args.load:
            model = WeirdNetwork.load(args.load)
        elif args.config:
            model = build_model(args.config)
        if not model:
            raise "I need to either --load a model or make a new one from a --config!"
        model = train(model, args.epochs, args.accuracy/100, args.graph)
        if args.save:
            print(f"saving model at {args.save}")
            model.save(args.save)
    elif args.action == "experiment":
        experiment(args.config, args.epochs)
    elif args.action == "run":
        model = None
        if args.load:
            model = WeirdNetwork.load(args.load)
        elif args.config:
            model = build_model(args.config)
        if not model:
            raise "I need to either --load a model or make a new one from a --config!"
        run(model, args.input)
    elif args.action == "play":
        play(args.config)