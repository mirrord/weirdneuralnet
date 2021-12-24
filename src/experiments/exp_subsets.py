
from weirdneuralnet.network import WeirdNetwork
from weirdneuralnet.subset_selection import *
from weirdneuralnet.datasets import get_dataset, get_accuracy
from tqdm import trange
import matplotlib.pyplot as plt
from pathlib import Path

def make_models(num_models, config, model_path=Path("./")):
    #TODO: check to make sure directory exists, create it if it doesn't
    for i in trange(num_models):
        m = WeirdNetwork.create_from_config(config)
        m.save(model_path / Path(f"models/model{i}.wm"))

def baseline(config, samples, training_type="normal", prime_epochs=1, epochs=30):
    #TODO: fix this lol
    print(f"constructing baseline with training type \"{training_type}\"...")
    results = []
    for i in trange(samples, desc="conducting experiments..."):
        X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')
        #model = build_model(config)
        model = WeirdNetwork.load(f"models\\model{i}.wm")
        epochs_done = 0
        if training_type == "primeA":
            new_X, new_Y = prime_typea(X_train, "kmeans", 10)
            cost_history = model.train(new_X, new_Y, prime_epochs)
            epochs_done = prime_epochs
        elif training_type == "primeB":
            new_X, new_Y = prime_typeb(X_train, "kmeans", 10)
            cost_history = model.train(new_X, new_Y, prime_epochs)
            epochs_done = prime_epochs
        elif training_type == "primeC":
            new_X, new_Y = prime_typec(X_train, Y_train, "kmeans", 10)
            cost_history = model.train(new_X, new_Y, prime_epochs)
            epochs_done = prime_epochs
        else:
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

def pretraining_experiment(config, samples):
    for train_type in ["primeC"]:
        for epochs in [50, 100]:
            if train_type != "normal":
                for prime_ratio in [0.1, 0.20, 0.5]:
                    baseline(config, samples, train_type, int(prime_ratio*epochs), epochs)
            else:
                baseline(config, samples, train_type, 1, epochs)