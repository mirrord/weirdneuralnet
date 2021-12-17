
import os
import hashlib, requests, gzip
import argparse
from tqdm import trange

import matplotlib.pyplot as plt

#import numpy as np
import cupy as np
from numpy import frombuffer, setdiff1d

from .network import WeirdNetwork, prime_typea, prime_typeb, prime_typec
from .node_utils import binarize

def get_dataset(path):
    #TODO: add more datasets, or a better way to include more
    # not really sure how best to do this yet
    path = os.path.join(os.path.join(os.path.dirname(__file__), '../..'), path)
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
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
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
    X_train = X_train.reshape((-1,28*28))
    X_val = X_val.reshape((-1,28*28))
    
    Y_train, Y_val, Y_test = binarize(Y_train, 10), binarize(Y_val, 10), binarize(Y_test, 10)
    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def get_accuracy(model, X, Y):
    prediction_classes = model.predict(X, debinarize=True)
    num_correct = np.where(np.equal(prediction_classes, Y.argmax(axis=1)))[0].shape[0]
    return num_correct, len(prediction_classes)

def train(model, epochs, acc_threshold, graph_it):
    #fetch data
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')

    print("train(): accuracy threshold not implemented yet")

    cost_history = model.train(X_train, Y_train, epochs)

    print(f"test error: {model.evaluate(X_test, Y_test)}")
    correct, total = get_accuracy(model, X_test, Y_test)
    print(f"total test samples: {total}\nnumber correct: {correct}\naccuracy: {correct/total}")

    if graph_it:
        plt.title("training progression")
        plt.ylabel("cost")
        plt.xlabel("epoch")
        plt.plot(list(range(epochs)), cost_history)
        plt.show()

    return model

def build_model(fname):
    import json
    with open(fname, 'r') as f:
        config = json.load(f)
    return WeirdNetwork.create_from_config(config)

def run(model, inp_fname):
    with open(inp_fname, 'rb') as f:
        #TODO: use a different ingestion method based on the type of file
        #   pull it into a np vector!
        buffer = f.read()
    return model.predict(buffer)

def baseline(config, samples, training_type="normal", prime_epochs=1, epochs=30):
    print(f"constructing baseline with training type \"{training_type}\"...")
    results = []
    for i in trange(samples, desc="conducting experiments..."):
        X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')
        #model = build_model(config)
        model = WeirdNetwork.load(f"models\\model{i}.wm")
        epochs_done = prime_epochs
        if training_type == "primeA":
            cost_history = prime_typea(model, X_train, "kmeans", 10, prime_epochs)
        elif training_type == "primeB":
            cost_history = prime_typeb(model, X_train, "kmeans", 10, prime_epochs)
        elif training_type == "primeC":
            cost_history = prime_typec(model, X_train, Y_train, "kmeans", 10, prime_epochs)
        else:
            epochs_done = 0
            cost_history = []
        cost_history.extend(model.train(X_train, Y_train, epochs-epochs_done))
        plt.plot(list(range(epochs)), cost_history)
        acc_train = get_accuracy(model, X_train, Y_train)
        acc_test = get_accuracy(model, X_test, Y_test)
        results.append(str((acc_train[0]/acc_train[1], acc_test[0]/acc_test[1])))

    feed="\n\t".join(results)
    with open(f"training_{training_type}_{epochs}epochs_{prime_epochs}prime.txt", "w") as f:
        f.write(f"training vs. validation results: \n\t{feed}")
    plt.title(f"{training_type} training progression")
    plt.ylabel("cost")
    plt.xlabel("epoch")
    path = "C:\\Users\\19082\\Desktop\\dev projects\\python\\ai\\experiment records\\big experiment"
    plt.savefig(f"{path}\\training_{training_type}_{epochs}epochs_{prime_epochs}prime.png")
    plt.close()

def experiment(config, samples=100):
    # for i in trange(samples):
    #     m = build_model(config)
    #     m.save(f"models/model{i}.wm")
    for train_type in ["primeC"]:
        for epochs in [50, 100]:
            if train_type != "normal":
                for prime_ratio in [0.1, 0.20, 0.5]:
                    baseline(config, samples, train_type, int(prime_ratio*epochs), epochs)
            else:
                baseline(config, samples, train_type, 1, epochs)
    # baseline(config, samples, "primeA", prime_epochs)
    # baseline(config, samples, "primeB", prime_epochs)
    # baseline(config, samples, "primeC", prime_epochs)
    

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
        play()