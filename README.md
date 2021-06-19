# Bacterial taxonomy Model

Author: Sonja Kittl
Disclaimer: This is a work in progress, it may therefore not work as expected.

## Introduction

Bacterial taxonomy is  largely based on 16SrRNA sequence clustering.
The idea of this project is to see if the neural network can learn to classify bacteria into orders based on 16SrRNA sequences.
As a test case I used 4 orders of actinobacteria:
- actinomycetales
- corynebacteriales (= mycobacteriales)
- micrococcales
- propionibacteriales

The network should be able to distinguish between these related bacterial orders based on 16SrRNA sequences.
Plublicly available data from genbank was used (including only RefSeq data): 

To get the data the following NCBI queries were used and data downloaded as fasta:

    ((33175[BioProject] OR 33317[BioProject])) AND txid2037[Organism] 

    ((33175[BioProject] OR 33317[BioProject])) AND txid85007[Organism] 

    ((33175[BioProject] OR 33317[BioProject])) AND txid85006[Organism]
    
    ((33175[BioProject] OR 33317[BioProject])) AND txid85009[Organism]

To replace the unique identifiers with uniform labels the following bash commands were used

- sed "/>/c>actinomycetales" txid2037.fasta > actinomycetales_taxid_la.fasta
- sed "/>/c>corynebacteriales" txid85007.fasta > corynebacteriales_taxid_la.fasta
- sed "/>/c>micrococcales" txid85006.fasta > micrococcales_taxid_la.fasta
- sed "/>/c>propionibacteriales" txid85009.fasta > propionibacteriales_taxid_la.fasta

merge into one dataset 
- cat *la.fasta > actinobacteria_data4.fasta

This datafile is provided as an example in the repository.

## Dependencies

- numpy
- matplotlib
- tensorflow
- re

## Usage

### Input

$ python3 bacteria_phylo.py yourdata.fasta epochs

Where yourdata.fasta is a fasta file containing your sequences with identifies that represent the desired classifiers. In the example data this is the genus. Epochs is the number of epochs you wish to train the model.

70% of the data of each group are used for training and 20% for validation and 10% for prediction.

### Output
Figure 1: plotting training and validation loss and accurracy
 
predict_results.txt: contains the predictions for the testing data (10% of input)

predict_data.fasta: The data used for prediction retransformed to fasta format with integer correponding to draw (same as in predict_results.txt) and text label.


