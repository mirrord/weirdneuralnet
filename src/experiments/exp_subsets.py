
from weirdneuralnet.network import WeirdNetwork
from weirdneuralnet.subset_selection import *
from weirdneuralnet.datasets import get_dataset, get_accuracy
from tqdm import trange
import matplotlib.pyplot as plt
from pathlib import Path

def make_models(num_models, config, model_path=Path("./")):
    model_path.mkdir(parents=True, exist_ok=True)
    for i in trange(num_models):
        m = WeirdNetwork.create_from_config(config)
        m.save(model_path / Path(f"models/model{i}.wm"))

def pretrain_and_train(samples, pretraining_type="normal", clutser_type="kmeans", prime_epochs=1, epochs=30):
    print(f"constructing baseline with training type \"{pretraining_type}/{clutser_type}\"...")
    results = [0]*(epochs+1)
    convergence_target = 0.9
    for i in trange(samples, desc="conducting experiments..."):
        X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')
        model = WeirdNetwork.load(f"models\\model{i}.wm")
        if pretraining_type == "primeA":
            new_X, new_Y = prime_typea(X_train, clutser_type, 10)
            cost_history = model.train(new_X, new_Y, prime_epochs)
        elif pretraining_type == "primeB":
            new_X, new_Y = prime_typeb(X_train, clutser_type, 10)
            cost_history = model.train(new_X, new_Y, prime_epochs)
        elif pretraining_type == "primeC":
            new_X, new_Y = prime_typec(X_train, Y_train, clutser_type, 10)
            cost_history = model.train(new_X, new_Y, prime_epochs)
        else:
            cost_history = []
        cost_history = model.train(X_train, Y_train, epochs, convergence_target)
        results[len(cost_history)] += 1
        plt.plot(list(range(len(cost_history))), cost_history)
    plt.plot(list(range(epochs+1)), results)
    plt.title(f"{pretraining_type}/{clutser_type} training time to convergence")
    plt.ylabel("number of models")
    plt.xlabel("epochs to converge")
    path = "C:\\Users\\19082\\Desktop\\dev projects\\python\\ai\\experiment records\\big experiment"
    plt.savefig(f"{path}\\training_{pretraining_type}_{clutser_type}_{prime_epochs}prime.png")
    plt.close()

def pretraining_experiment(samples):
    for pretrain_type in ["normal", "primeB", "primeC"]:
        if pretrain_type != "normal":
            for cluster_type in CLUSTER_FUNCS.keys():
                for epochs in [10, 20, 50]:
                    pretrain_and_train(samples, pretrain_type, cluster_type, epochs, 300)
        else:
            pretrain_and_train(samples, "normal", "", 1, 300)