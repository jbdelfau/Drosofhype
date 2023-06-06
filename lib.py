#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor, RegressorChain

random.seed()
np.random.seed()


class Drosophype:
    """
    DROSOFHYPE: Directed, Rapid and Optimized Search Of Fitting HYPErparameters.
    Genetic algorithm that searches for the optimal hyperparameters of a machine-learning model.
    """

    def __init__(self, model, config, multioutput=False, multioutput_type=None, initial_individuals=None):
        """
        Initializes class variables as well as the initial generation.
        :param model: model for which we seek the optimal hyperparameters
        :param config: configuration of the search process:
            - population size: size of each generation
            - n_parents: number of best performing individuals kept to generate children
            - mutation_rate: occurence of children mutations(between 0 and 1)
            - additional_properties: properties to add to the output results
            - hyperparameters: characteristics of the hyperparameters we want to optimize
            (see jsonschema for more information)
        :param multioutput: is the model using a scikit-learn multi target function such as
            sklearn.multioutput.MultiOutputRegressor
        :param multioutput_type: type of multi target function (only relevant if multioutput is set to True)
        :param initial_individuals: specific population for the first generation

        :type model: machine-learning algorithm object
        :type config: nested dict
        :type multioutput: bool
        :type multioutput_type: str
        :type initial_individuals: list of dict
        """

        self.model = model
        self.multioutput = multioutput
        self.chained_multioutput = multioutput_type
        self.config = config
        self.initial_individuals = initial_individuals
        # Create initial population
        self.generation = 0
        self.population = self.init_population()

    def crossovers(self, parents):
        """
        Creates children by randomly selecting characteristics of the parents.
        :param parents: parents
        :return: children

        :type parents: list of dict
        :rtype: list of dict
        """

        children = []
        for i in range(self.config["population_size"] - self.config["n_parents"]):
            child = {param: self.pick_parents_attribute(parents, param) for param in
                     self.config["hyperparameters"].keys()}
            children.append(child)

        return children

    def init_population(self):
        """
        Creates the initial generation.
        :return: individuals of the initial generation

        :rtype: list of dict
        """

        if self.initial_individuals is not None:
            population = self.initial_individuals
            _size = self.config["population_size"] - len(self.initial_individuals)
        else:
            population = []
            _size = self.config["population_size"]

        for i in range(_size):
            individual = {"already_evaluated": False}
            for attr in self.config["hyperparameters"].keys():
                _param = self.config["hyperparameters"][attr]
                if _param["type"] == "float":
                    individual[attr] = round(random.uniform(_param["min"], _param["max"]), _param["round"])
                elif _param["type"] == "int":
                    individual[attr] = random.randint(_param["min"], _param["max"])
                elif _param["type"] == "constant":
                    individual[attr] = _param["value"]
                elif _param["type"] == "category":
                    individual[attr] = random.choice(_param["possible_values"])
            population.append(individual)

        return population

    def find_best_params(self, train_data, train_labels, test_data, test_labels):
        """
        Runs one generation of genetic search by:
            1. training the model on the training data for each individual of the current population
            2. evaluating each individual and keeping only the best ones (called "parents")
            3. creating new individuals ("children") from crossovers between the parents
            4. adding mutations to the children
            5. defining a new population including parents and children

        :param train_data: training data
        :param train_labels: training labels
        :param test_data: testing data
        :param test_labels: testing labels
        :return: characteristics and performances of the individuals of this generation

        :type train_data: pandas.core.frame.DataFrame
        :type train_labels: numpy.ndarray
        :type test_data: pandas.core.frame.DataFrame
        :type train_labels: numpy.ndarray
        :rtype: pandas.core.frame.DataFrame
        """

        # Training model for each individual
        self.population = self.train_model_for_population(self.population, train_data, train_labels, test_data,
                                                          test_labels)
        # Selecting best individuals
        _parents = self.select_best_individuals(self.population)
        # Creating new individuals by mixing best ones properties
        _children = self.crossovers(_parents)
        # Adding mutations
        _mutated_children = self.mutations(_children)
        # Export population results
        _results = self.format_population_results(self.population, self.generation)
        _parents = self.mark_as_evaluated(_parents)
        # Defining new population from parents and children
        self.population = _parents + _mutated_children
        self.generation += 1

        return _results

    def format_population_results(self, population, generation):
        """
        Formats results of the current generation.
        :param population: population of the generation
        :param generation: generation number
        :return: results of the current generation

        :type population: list of dict
        :type generation: int
        :rtype: pandas.core.frame.DataFrame
        """

        _keys_to_keep = [k for k in population[0].keys() if k != "already_evaluated"]
        _to_export = [x for x in population if not x["already_evaluated"]]
        results_data = pd.DataFrame({attr: [x[attr] for x in _to_export] for attr in _keys_to_keep})
        for attr, val in self.config["additional_properties"]:
            results_data[attr] = val
        results_data.insert(0, "generation", generation)
        return results_data

    def evaluate_performances(self):
        """
        Evaluates the performances of a model trained with the parameters of an individual.
        This function has to be defined by the user as its depends on the use case.
        """
        pass

    def mark_as_evaluated(self, parents):
        """
        Labels parents as already evaluated to avoid losing time with evaluation again.
        :param parents: parents
        :return:parents

        :type parents: list of dict
        :rtype: list of dict
        """

        for x in parents:
            x["already_evaluated"] = True
        return parents

    def mutate_attribute(self, attribute, val):
        """
        Mutates the value of a given attribute i.e. changes slightly its value.
        :param attribute: name of the aatribute to which we apply mutation
        :param val: value of the attribute
        :return: new value of the attribute

        :type attribute: str
        :type val: int or float
        :rtype: int or float
        """

        p = self.config["hyperparameters"][attribute]
        if p["type"] == "float":
            delta = np.random.uniform(p["range"][0], p["range"][1])
            return min(p["max"], max(p["min"], round(val + delta, p["round"])))
        elif p["type"] == "int":
            delta = np.random.randint(p["range"][0], p["range"][1])
            return min(p["max"], max(p["min"], val + delta))
        elif p["type"] == "category":
            return random.choice(p["possible_values"])
        elif p["type"] == "constant":
            return p["value"]

    def mutations(self, children):
        """
        Mutates the population of children by randomly modifying the values of their attributes.
        :param children: population of children
        :return: children with mutations

        :type children: list of dict
        :rtype: list of dict
        """

        x_children = []
        for child in children:
            x_child = {attribute: self.mutate_attribute(attribute, val) if np.random.uniform() <= self.config[
                "mutation_rate"] else val for attribute, val in child.items()}
            x_child["already_evaluated"] = False
            x_children.append(x_child)

        return x_children

    def pick_parents_attribute(self, parents, attribute):
        """
        Randomly selects the required attribute from one of the parents. The selection is not purely random but
        pseudo-random: parents with higher performances have higher chances to be selected.
        :param parents: parents
        :param attribute: name of the attribute to pick
        :return: value of the required attribute

        :type parents: list of dict
        :type attribute: str
        :rtype: int or float
        """

        tot_perfs = sum([x["performances"] for x in parents])
        assert tot_perfs != 0
        boundaries = [x["performances"] / tot_perfs for x in parents]
        boundaries = np.cumsum(boundaries)

        rand = np.random.uniform()
        i = 0
        while i < len(boundaries):
            if rand <= boundaries[i]:
                return parents[i][attribute]
            else:
                i += 1

        assert True, "erreur"

    def select_best_individuals(self, population):
        """
        Returns the required number of best performing individuals.
        :param population: total population of the generation
        :return: best performing individuals

        :type population: list of dict
        :rtype: list of dict
        """

        chosen_ones = sorted(population, key=lambda x: x["performances"])[:self.config["n_parents"]]
        return chosen_ones

    def train_model_for_population(self, population, train_data, train_labels, test_data, test_labels):
        """
        Trains the model for each individual of the generation and evaluates its performances.
        :param population: individuals of the current generation
        :param train_data: training data
        :param train_labels: training labels
        :param test_data: test data
        :param test_labels: test labels
        :return: individuals with their performances

        :type population: list of dict
        :type train_data: pandas.core.frame.DataFrame
        :type train_labels: numpy.ndarray
        :type test_data: pandas.core.frame.DataFrame
        :type train_labels: numpy.ndarray
        :rtype: list of dict
        """

        for individual in population:
            if not individual["already_evaluated"]:
                _params = {k: v for k, v in individual.items() if k != "already_evaluated"}
                if self.multioutput:
                    if self.chained_multioutput == "RegressorChain":
                        _model = RegressorChain(self.model(**_params)).fit(train_data, train_labels)
                    elif self.chained_multioutput == "MultiOutputRegressor":
                        _model = MultiOutputRegressor(self.model(**_params)).fit(train_data, train_labels)
                    else:
                        raise Exception("The multioutput function to use has to be spectified.")
                else:
                    _model = self.model(**_params).fit(train_data, train_labels)

                preds = _model.predict(test_data)
                individual["performances"] = self.evaluate_performances(test_labels, preds)

        return population
