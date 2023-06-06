======================
 DROSOFHYPE
======================

Directed, Rapid and Optimized Search Of Fitting HYPErparameters.


* GitHub repo: https://github.com/jbdelfau/Drosofhype
* Free software: GNU license

Features
--------

A genetic algorithm that searches for the optimal hyperparameters of a machine-learning model. 


Quickstart
----------

This library requires a machine-learning model and a json specifying the genetic search configuration. 2 examples are provided in the files <em>test_svm.py</em> and <em>test_xgbregressor.py</em>. 

Let's start with the parameters of the genetic search:
- **population_size**: size of each generation.
- **n_parents**: number of best performing individuals kept to generate children.
- **mutation_rate**: occurence of children mutations(between 0 and 1).
- **additional_properties**: information that must be included in the output table as additional columns.

In addition, the json must contain an additional key - <em>hyperparameters</em> - describing which parameters of the model have to be varied and in what range. See the above files for more examples.