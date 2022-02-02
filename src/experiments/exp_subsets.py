
from weirdneuralnet.network import WeirdNetwork
from weirdneuralnet.subset_selection import *
from weirdneuralnet.datasets import get_dataset, get_accuracy
from tqdm import trange
import matplotlib.pyplot as plt
from pathlib import Path
import json

def make_models(num_models, config, model_path=Path("./")):
    model_path.mkdir(parents=True, exist_ok=True)
    for i in trange(num_models):
        m = WeirdNetwork.create_from_config(config)
        m.save(model_path / Path(f"models/model{i}.wm"))

def pretrain_and_train(samples, preX, preY, pretraining_type="normal", clutser_type="kmeans", prime_epochs=1, epochs=30):
    print(f"constructing baseline with training type \"{pretraining_type}/{clutser_type}\"...")
    results = [0]*(epochs+1)
    convergence_target = 0.9
    for i in trange(samples, desc="conducting experiments..."):
        X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')
        model = WeirdNetwork.load(f"models\\model{i}.wm")
        if pretraining_type != "normal":
            cost_history = model.train(preX, preY, prime_epochs)
        else:
            cost_history = []
        cost_history = model.train(X_train, Y_train, epochs, convergence_target)
        results[len(cost_history)] += 1
        #plt.plot(list(range(len(cost_history))), cost_history)
    plt.plot(list(range(epochs+1)), results)
    plt.title(f"{pretraining_type}/{clutser_type} training time to convergence")
    plt.ylabel("number of models")
    plt.xlabel("epochs to converge")
    path = "C:\\Users\\19082\\Desktop\\dev projects\\python\\ai\\experiment records\\big experiment"
    plt.savefig(f"{path}\\training_{pretraining_type}_{clutser_type}_{prime_epochs}prime.png")
    plt.close()

def pretraining_experiment(samples):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')
    for pretrain_type, pretrain_func in {"normal":None, "primeB":lambda x: prime_typeb(X_train, x, 10), "primeC":lambda x: prime_typec(X_train, Y_train, x, 10)}.items():
        if pretrain_type != "normal":
            for cluster_type in CLUSTER_FUNCS.keys():
                if pretrain_type=="primeB" and cluster_type=="kmeans": continue
                pretrain_X, pretrain_Y = pretrain_func(cluster_type)
                for epochs in [10, 20, 50]:
                    pretrain_and_train(samples, pretrain_X, pretrain_Y, pretrain_type, cluster_type, epochs, 1000)
        else:
            pass
            #pretrain_and_train(samples, "normal", "", 1, 300)

##### new experimental structure

def _pretraining_cache(far_points=1, nested_clusters=3):
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')
    primer_funcs = {"primeB":lambda x: (prime_typeb(X_train, x, 10, far_points), far_points ), 
                    "primeB2":lambda x: (prime_typeb(X_train, x, 10, (nested_clusters*(far_points+1))-1 ), (nested_clusters*(far_points+1))-1),
                    "primeC":lambda x: (prime_typec(X_train, Y_train, x, 10, nested_clusters, far_points), nested_clusters*far_points)}
    cache = {}
    for cluster_type in ["kmeans"]: #CLUSTER_FUNCS.keys():
        for prime_type, pfunc in primer_funcs.items():
            subset, num_points = pfunc(cluster_type)
            cache[f"{prime_type}_{cluster_type}_{num_points}"] = subset
    return cache

def create_cached_models(samples):
    pretrain_cache = _pretraining_cache()
    # for each model, create a profile of pretrained models to start from
    for model_idx in trange(samples, desc="creating models..."):
        model = WeirdNetwork.load(f"models\\model{model_idx}.wm")
        for key, trainset in pretrain_cache.items():
            for prepochs in range(10):
                model.train(trainset[0], trainset[1], 20)
                model.save(f"cached_models\\model{model_idx}_{prepochs*20}_{key}.wm")

def get_cached_models():
    flist = [p for p in Path('cached_models').iterdir() if p.is_file()]
    for fname in flist:
        yield fname, WeirdNetwork.load(fname)
    return

def _add_stats(s):
    with open("stats.txt", 'a') as f:
        f.write(json.dumps(s)+"\n")

def training_average(samples):
    max_epochs = 2000
    convergence_target = 0.9
    total_conv = 0
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')
    for model_idx in trange(samples, desc="creating models..."):
        model = WeirdNetwork.load(f"models\\model{model_idx}.wm")
        cost_history = model.train(X_train, Y_train, max_epochs, convergence_target, 5000)
        total_conv+=len(cost_history)
    print(f"average epochs to convergence: {total_conv/100}")

def pretraining_exp_huge(samples):
    max_epochs = 2000
    convergence_target = 0.9
    X_train, Y_train, X_test, Y_test, X_val, Y_val = get_dataset('datasets')
    # for each model, create a profile of pretrained models to start from
    for fname, model in get_cached_models():
        params = str(fname).split('_')
        stats = {
            "index": params[0][params[0].find('model')+5:],
            "pretrain epochs": params[1],
            "pretraining type": params[2],
            "cluster type": params[3],
            "subset size": params[4][:params[4].find('.')]
        }
        cost_history = model.train(X_train, Y_train, max_epochs, convergence_target, 5000)
        stats['convergence time'] = len(cost_history)+1
        _add_stats(stats)
    return