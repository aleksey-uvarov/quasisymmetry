#! /bin/bash

python find_quartets.py hamiltonians/H2O_OH0.9580_104.5000.chk
python find_quartets.py hamiltonians/H2O_OH1.6430_104.5000.chk
python find_quartets.py hamiltonians/H2O_OH2.5000_104.5000.chk

python find_quartets.py hamiltonians/H2O_OH0.9580_104.5000.chk --optimization_mode OO --quartet_graph matching
python find_quartets.py hamiltonians/H2O_OH1.6430_104.5000.chk --optimization_mode OO --quartet_graph matching
python find_quartets.py hamiltonians/H2O_OH2.5000_104.5000.chk --optimization_mode OO --quartet_graph matching

python find_quartets.py hamiltonians/H2O_OH0.9580_104.5000.chk --optimization_mode OO --quartet_graph ring
python find_quartets.py hamiltonians/H2O_OH1.6430_104.5000.chk --optimization_mode OO --quartet_graph ring
python find_quartets.py hamiltonians/H2O_OH2.5000_104.5000.chk --optimization_mode OO --quartet_graph ring
