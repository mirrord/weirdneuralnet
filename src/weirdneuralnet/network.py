from .node import DelayNode, DualInputNode, NeuralNode
from .node_utils import *
from .cluster import *

# import numpy as np
import cupy as np
from tqdm import trange
from pathlib import Path
import json

from pickle import Pickler, Unpickler


# TODO still need to test this
# ...and it's going to need a huge upgrade
class CompiledNetwork:
    """a topology-agnostic gradient descent NN with lower overhead."""

    def __init__(
        self,
        weights,
        biases,
        edges,
        activations,
        normalizations,
        input_idxs=[0],
        output_idx=-1,
    ):
        self.weights = weights
        self.biases = biases
        self.edges = edges
        self.activation_lables = activations
        self.activations = [
            ACTIVATIONS.get(act, (sigmoid, dsigmoid)) for act in activations
        ]
        self.normalization_params = normalizations
        self.normalizations = [
            NORMALIZATIONS.get(norm, nonorm) for norm in normalizations
        ]
        self.inputs = {}
        self.outputs = {}
        self.zs = {}
        self.feed_indices = [
            [edge[1] for edge in edges if edge[0] == idx] for idx in range(len(weights))
        ]
        self.backfeed_indices = [
            [edge[0] for edge in edges if edge[1] == idx] for idx in range(len(weights))
        ]
        self.input_nodes = input_idxs
        self.output_node = output_idx if output_idx != -1 else len(self.nodes) - 1

    # TODO: implement compiled training

    def predict(self, input: np.array):
        """Calculate this model's prediction for some input.
        Inputs
            input
                the input vector. It must be of shape (features, samples).
        """
        outputs = {}
        to_traverse = []
        for idx in self.input_indices:
            self.inputs[idx] = input
            if self.normalize_label:
                self.inputs[idx] = self.normalize(input)
            self.zs[idx] = np.dot(self.inputs[idx], self.weights[idx]) + self.bias[idx]
            self.outputs[idx] = self.activations[idx][0](self.zs[idx])
            to_traverse.extend(self.feed_indices[idx])
        for idx in to_traverse:
            if idx not in outputs:
                # find outputs this node wants as input
                inputs = [
                    self.nodes[i].get_output() for i in self.backfeed_indices[idx]
                ]
                # "synapse" them together
                self.inputs[idx] = sum(inputs)
                if self.normalize_label:
                    self.inputs[idx] = self.normalize(self.inputs[idx])
                self.zs[idx] = (
                    np.dot(self.inputs[idx], self.weights[idx]) + self.bias[idx]
                )
                self.outputs[idx] = self.activations[idx][0](self.zs[idx])
                to_traverse.extend(
                    [fidx for fidx in self.feed_indices[idx] if fidx not in outputs]
                )
        if self.output_node in self.outputs:
            return self.outputs[self.output_node]
        raise Exception("Output node is not fed")

    def backprop(
        self,
        input,
        expout,
        derror_func=ddiff_squares,
    ):
        """Calculate weight & bias updates using gradient descent.
        Inputs
            input
                vector on which to calculate error singal, shape (features, samples).
            exp_output
                vector to compare with model prediction, shape (binarized classes, samples).
        """
        num_sample = input.shape[1]
        backfeed = {}
        de_dz = derror_func(self.predict(input), expout)
        backfeed[self.output_node][0] = de_dz * self.activations[self.output_node][1](
            self.zs[self.output_node]
        )
        backfeed[self.output_node][1] = np.dot(
            self.inputs[self.output_node].T, backfeed[self.output_node][0]
        )
        backfeed[self.output_node][2] = np.dot(
            backfeed[self.output_node][0], self.weights[self.output_node].T
        )
        to_traverse = self.backfeed_indices.get(self.output_node, []).copy()
        for idx in to_traverse:
            error_signal_components = [
                backfeed.get(oidx, (0, 0, 0)) for oidx in self.feed_indices[idx]
            ]
            de = sum([i[2] for i in error_signal_components])
            db = de_dz * self.activations[idx][1](self.zs[idx])
            dw = np.dot(self.inputs[idx].T, backfeed[idx][0])
            de = np.dot(backfeed[idx][0], self.weights[idx].T)
            db, dw, de = self.nodes[idx].backfeed(de)
            if idx not in backfeed:
                backfeed[idx] = db, dw, de
            else:
                backfeed[idx][0] += db
                backfeed[idx][1] += dw
                backfeed[idx][2] += de

            for jdx in self.backfeed_indices[idx]:
                if jdx not in to_traverse:
                    to_traverse.append(jdx)

        bup = {
            i: np.sum(b[0], 0, keepdims=True) / num_sample for i, b in backfeed.items()
        }
        wup = {i: w[1] / num_sample for i, w in backfeed.items()}
        return bup, wup


############################################################################################################
### Network V2, under construction
############################################################################################################


class WeirdNetwork:
    """A gradient descent NN of arbitrary topology.
    Methods
    --------
    load (classmethod)
        load a network instance from a saved model file.
    save
        save a network to a file.
    compile
        compile this network into a raw network with less overhead.
    predict
        compute forward propagation on some input.
    get_last_output
        retrieve the last-calculated output of this network.
    backpropagate
        calculate weight & bias updates based on input & expected output.
    evaluate
        compute the error cost of the network's prediction on some input & expected output.
    train
        train the network on some set of inputs and expected outputs for some number of epochs."""

    def __init__(
        self,
        node_params,
        edges,
        error_cost_func="diff_squares",
        learning_rate=0.01,
        regularize=(None, 0),
    ):
        """Construct a WeirdNetwork NN.
        Inputs
            node_params
                A list containing a dict for each node in the network. Each dict
                must include at minimum: "x" (input dimension), "y" (output dimension),
                and "activation".
            edges
                a list of (int,int) tuples that describe the topology of the network
                as a directed graph.
            error_cost_func
                a string label for the error cost function to be used.
            learning_rate
                the learning coefficient to be used during training.
            regularize
                a (str, int...) tuple that contains the string label for the
                regularization function to be used during training and its parameters.
        """
        # node_params should be a list of dicts of node parameters
        # edges should be a list of node index tuples denoting connections between nodes
        self.nodes = []
        self.input_indices = []
        # which nodes feed into which input sites
        self.feed_indices = {}
        # which input sites take input from which nodes
        self.backfeed_indices = {}
        self.output_node = 0
        self._stepwise_training = False
        self.learning_rate = learning_rate
        self.input, self.output = None, None
        self.regularize_params, self.regularize = regularize, REGULARIZATIONS.get(
            regularize[0], noreg
        )(regularize[1])
        self.cost, self.cost_deriv = COSTS.get(
            error_cost_func, (diff_squares, ddiff_squares)
        )
        self.edges = [
            edge if len(edge) == 3 else (edge[0], edge[1], 0) for edge in edges
        ]
        for idx, param in enumerate(node_params):
            self.feed_indices[idx] = [
                (edge[1], edge[2]) for edge in self.edges if edge[0] == idx
            ]
            for outp, inp, site in self.edges:
                if inp == idx:
                    self.backfeed_indices[(idx, site)] = [
                        edge[0]
                        for edge in self.edges
                        if edge[1] == idx and edge[2] == site
                    ]
            node_type = param.get("type", "neural")
            if node_type == "neural":
                self.nodes.append(
                    NeuralNode(
                        param["x"],
                        param["y"],
                        param.get("activation", ""),
                        param.get("normalize", ""),
                    )
                )
            elif node_type == "dual-input":
                self.nodes.append(
                    DualInputNode(
                        param.get("activation", "scaled"), param.get("normalize", "")
                    )
                )
            elif node_type == "delay":
                self.nodes.append(DelayNode())
                self._stepwise_training = True
            else:
                raise NotImplementedError(f"node type {node_type} not recognized")
            if param.get("output", False):
                self.output_node = idx
            if param.get("input", False):
                self.input_indices.append(idx)

    def _profile(self):
        input_dim = None
        for i in range(self.input_indices):
            shp = self.nodes[i].shape()
            if shp != None:
                input_dim = shp[0]
                break
        if input_dim == None:
            raise Exception("No suitable input node!")
        input_vec = np.ones((1, input_dim))
        outputs = {}
        dims = {idx: (0, 0) for idx in range(len(self.nodes))}
        to_traverse = []
        for idx in self.input_indices:
            try:
                outputs[idx] = self.nodes[idx].feed({0: [input_vec]})
                dims[idx] = (input_vec.shape[1], outputs[idx].shape[1])
                if self.nodes[idx].shape() == None:
                    self.nodes[idx].set_dims(dims[idx])
            except:  # TODO: what is the matmult exception?
                raise Exception(f"input node ID {idx} has a problem!")
            to_traverse.extend(self.feed_indices[idx])
        for idx, _ in to_traverse:
            if idx not in outputs:  # if node has not yet been fired
                # find outputs this node wants as input
                inputs = {
                    site: [
                        self.nodes[i].get_output()
                        for i in self.backfeed_indices[(idx, site)]
                        if self.nodes[i].get_output() is not None
                    ]
                    for site in range(self.nodes[idx].num_input_sites)
                }
                if all(
                    [
                        len(inputs[site]) == len(self.backfeed_indices[(idx, site)])
                        for site in range(self.nodes[idx].num_input_sites)
                    ]
                ):
                    # fire iff: all sites filled, all edges satisfied
                    # print(f"feeding node {idx} with input:\n{inputs}")
                    outputs[idx] = self.nodes[idx].feed(inputs)
                    to_traverse.extend(
                        [fidx for fidx in self.feed_indices[idx] if fidx not in outputs]
                    )
        if self.output_node in outputs:
            return True
        raise Exception("Output node is not fed")

    def __str__(self):
        return f"<WeirdNetwork size={len(self.nodes)} nodes>"

    @classmethod
    def load(cls, fname: str):
        """Load a WeirdNetwork model instance from a file."""
        with open(fname, "rb") as f:
            u = Unpickler(f)
            model = u.load()
        for node in model.nodes:
            node.reload()
        model.regularize = REGULARIZATIONS.get(model.regularize_params[0], noreg)(
            model.regularize_params[1]
        )
        return model

    @classmethod
    def create_from_config(cls, fname: Path):
        with open(fname, "r") as f:
            config = json.load(f)
        return cls.create_from_json(config)

    @classmethod
    def create_from_json(cls, json_config):
        return cls(
            json_config["node_params"],
            json_config["edges"],
            json_config.get("cost function", "diff_squares"),
            json_config.get("learning rate", 0.01),
            (
                json_config.get("regularization function", None),
                json_config.get("regularization parameter", 0),
            ),
        )

    def save(self, fname: str):
        """Save a WeirdNetowrk model instance to a file."""
        self.regularize = None
        for node in self.nodes:
            node.clear_history()
        with open(fname, "wb") as f:
            p = Pickler(f)
            p.dump(self)
        for node in self.nodes:
            node.reload()
        self.regularize = REGULARIZATIONS.get(self.regularize_params[0], noreg)(
            self.regularize_params[1]
        )

    def get_last_output(self):
        """Retrieve the last-calculated model output."""
        return self.nodes[self.output_node].output

    def evaluate(self, input: np.array, exp_out: np.array):
        """Calculate cost for some input & expected output."""
        return self.cost(self.predict(input), exp_out)

    def predict(self, input: np.array, debinarize=False):
        """Calculate this model's prediction for some input.
        Inputs
            input
                the input vector. It must be of shape (features, samples).
            debinarize
                if set to True, return the output as a classifier integer
                instead of a vector.
        """
        fired = []
        to_traverse = []
        for idx in self.input_indices:
            # print(f"feeding input node {idx}")
            self.nodes[idx].feed({0: [input]})
            fired.append(idx)
            to_traverse.extend(self.feed_indices[idx])
        for idx, _ in to_traverse:
            if idx not in fired:  # if node has not yet been fired
                # find outputs this node wants as input
                inputs = {
                    site: [
                        self.nodes[i].get_output()
                        for i in self.backfeed_indices[(idx, site)]
                        if self.nodes[i].get_output() is not None
                    ]
                    for site in range(self.nodes[idx].num_input_sites)
                }
                if all(
                    [
                        len(inputs[site]) == len(self.backfeed_indices[(idx, site)])
                        for site in range(self.nodes[idx].num_input_sites)
                    ]
                ):
                    # fire iff: all sites filled, all edges satisfied
                    # print(f"feeding node {idx} with input:\n{inputs}")
                    self.nodes[idx].feed(inputs)
                    fired.append(idx)
                    to_traverse.extend(
                        [fidx for fidx in self.feed_indices[idx] if fidx not in fired]
                    )
        if self.output_node in fired:
            if debinarize:
                return self.nodes[self.output_node].get_output().argmax(axis=1)
            return self.nodes[self.output_node].get_output()
        raise Exception("Output node is not fed")

    def backpropagate(self, exp_output: np.array):
        """Calculate weight & bias updates using gradient descent.
        Inputs
            input
                vector on which to calculate error signal, shape (features, samples).
            exp_output
                vector to compare with model prediction, shape (binarized classes, samples).
        """
        num_sample = exp_output.shape[0]
        predicted_output = self.nodes[self.output_node].output
        backfeed = {}  # node index: (db, dw, [ error signals by input site ])
        error_delta = self.cost_deriv(predicted_output, exp_output)
        backfeed[self.output_node] = self.nodes[self.output_node].backfeed(error_delta)
        to_traverse = []
        for site in range(self.nodes[self.output_node].num_input_sites):
            to_traverse.extend(
                self.backfeed_indices.get(((self.output_node, site)), []).copy()
            )

        for idx in to_traverse:
            error_signal_components = [
                backfeed.get(oidx, (0, 0, [0] * (site + 1)))[2][site]
                for oidx, site in self.feed_indices[idx]
            ]
            de = sum([i for i in error_signal_components])
            db, dw, de = self.nodes[idx].backfeed(de)
            backfeed[idx] = db, dw, de

            for site in range(self.nodes[idx].num_input_sites):
                for jdx in self.backfeed_indices.get((idx, site), []):
                    if jdx not in to_traverse:
                        to_traverse.append(jdx)

        bup = {
            i: np.sum(b[0], 0, keepdims=True) / num_sample
            for i, b in backfeed.items()
            if self.nodes[i].does_update()
        }
        wup = {
            i: w[1] / num_sample
            for i, w in backfeed.items()
            if self.nodes[i].does_update()
        }
        return bup, wup

    def train(
        self,
        input: np.array,
        exp_output: np.array,
        epochs: int,
        convergence_target: float = 1,
        batch_size: int = -1,
        display_progress_bar=True,
    ):
        """train the network on some set of inputs and expected outputs for some number of epochs."""
        # TODO: use test set to evaluate performance, not training set cost?
        # research this better
        mrange = lambda t: trange(t, desc="training...")
        mrange = mrange if display_progress_bar else range
        predict = lambda x: self.predict
        cost_history = []
        num_samples = len(exp_output)
        batch_size = num_samples if batch_size == -1 else batch_size
        for i in mrange(epochs):
            shuffle_in_unison(input, exp_output)
            cur_idx = 0
            accuracy = 0
            while cur_idx < num_samples:
                end_idx = cur_idx + batch_size
                predicted_output = self.predict(input[cur_idx:end_idx])
                accuracy = (
                    np.where(
                        np.equal(
                            debinarize(predicted_output),
                            exp_output[cur_idx:end_idx].argmax(axis=1),
                        )
                    )[0].shape[0]
                    / batch_size
                )
                bup, wup = self.backpropagate(exp_output[cur_idx:end_idx])
                # update
                for idx, node in enumerate(self.nodes):
                    if idx in wup:
                        update_weights = self.learning_rate * wup[idx]
                        if self.regularize_params[0] is not None:
                            update_weights += self.regularize(wup)
                        node.update(self.learning_rate * bup[idx], update_weights)
                cur_idx += batch_size
            cost_history.append(
                self.cost(
                    self.get_last_output(), exp_output[cur_idx - batch_size : end_idx]
                )
                / batch_size
            )
            if accuracy >= convergence_target:
                break

        return cost_history
