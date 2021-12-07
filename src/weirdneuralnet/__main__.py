
import os
import hashlib, requests, gzip
import argparse

import matplotlib.pyplot as plt

#import numpy as np
import cupy as np
from numpy import frombuffer, setdiff1d

from .network import WeirdNetwork
from .node_utils import binarize

def get_dataset(path):
    def fetch(url):
        fp = os.path.join(path, hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                data = f.read()
        else:
            with open(fp, "wb") as f:
                data = requests.get(url).content
                f.write(data)
        #return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()) #numpy version
        return np.array(frombuffer(gzip.decompress(data), dtype=np.uint8).copy())

    X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28)).T
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    #Validation split
    rand=np.arange(60000)
    np.random.shuffle(rand)
    train_no=rand[:50000]

    #val_no=np.setdiff1d(rand,train_no) #numpy version
    val_no=np.array(setdiff1d(np.asnumpy(rand),np.asnumpy(train_no)))

    X_train,X_val=X[train_no,:,:],X[val_no,:,:]
    Y_train,Y_val=Y[train_no],Y[val_no]
    #reshape
    X_train = X_train.reshape((-1,28*28)).T
    X_val = X_val.reshape((-1,28*28)).T
    
    Y_train, Y_val, Y_test = binarize(Y_train, 10).T, binarize(Y_val, 10).T, binarize(Y_test, 10).T
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def get_accuracy(model, X, Y):
    prediction_classes = model.predict(X, debinarize=True)
    num_correct = np.where(np.equal(prediction_classes, Y.argmax(axis=0)))[0].shape[0]
    return num_correct, len(prediction_classes)

def train(model, epochs, acc_threshold, graph_it):
    #fetch data
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset(r'C:\Users\19082\Desktop\dev projects\python\ai\weirdneuralnet\datasets')

    print("train(): accuracy threshold not implemented yet")

    cost_history = model.train(X_train, Y_train, epochs)

    print(f"test error: {model.evaluate(X_test, Y_test)}")
    correct, total = get_accuracy(model, X_test, Y_test)
    print(f"total test samples: {total}\nnumber correct: {correct}\naccuracy: {correct/total}")

    if graph_it:
        plt.plot(list(range(epochs)), cost_history)
        plt.show()

    return model

def build_model(fname):
    # import json
    # with open(fname, 'r') as f:
    #     config = json.load(f.read())
    print("build_model is not implemented yet")
    node_params =[
        {
            'x':28*28,
            'y':128,
            'activation': 'sigmoid',
            'input':True
        },
        {
            'x':128,
            'y':128,
            'activation': 'sigmoid',
        },
        {
            'x':128,
            'y':10,
            'activation': 'sigmoid',
            'output':True
        }
    ]
    edges = [
        (0,1),
        (0,2),
        (1,2)
    ]
    return WeirdNetwork(node_params, edges)

def run(model, inp_fname):
    with open(inp_fname, 'rb') as f:
        #TODO: use a different ingestion method based on the type of file
        #   pull it into a np vector!
        buffer = f.read()
    return model.predict(buffer)

def experiment():
    print("not implemented yet")

def play():
    print("this space reserved for *really* weird experiments")

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
        experiment()
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
        play()