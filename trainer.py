
import pickle
import random
import os
from typing import List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import ticker


import torch
from transformer import *

def train_model(n_encoder_blocks,n_decoder_blocks,n_heads):
    train_sentences, test_sentences, source_vocab, target_vocab = load_data()

    max_length = 64
    # Generates train/test data based on english and french vocabulary sizes and caps max length of sentence at 64
    train_source, train_target = preprocess_data(train_sentences, len(source_vocab), len(target_vocab), max_length)
    test_source, test_target = preprocess_data(test_sentences, len(source_vocab), len(target_vocab), max_length)

    embedding_dim = 256
    model = Transformer(len(source_vocab),len(target_vocab),embedding_dim,n_encoder_blocks,n_decoder_blocks,n_heads)
    epoch_train_loss, epoch_test_loss = train(model, train_source, train_target,test_source, test_target, len(target_vocab))

    torch.save(model.state_dict(),f'saved_weights_{n_encoder_blocks}_{n_decoder_blocks}_{n_heads}.pkl')
    np.save(f'train_losses_{n_encoder_blocks}_{n_decoder_blocks}_{n_heads}',epoch_train_loss)
    np.save(f'test_losses_{n_encoder_blocks}_{n_decoder_blocks}_{n_heads}',epoch_test_loss)
    print('train_loss',epoch_train_loss[-1])
    print('test_loss',epoch_test_loss[-1])



def compute_bleu_score(model,test_source,test_target):
    assert len(test_source) == len(test_target)
    beam_size= 3
    total_score = 0

    for i in range(len(test_source)):
        predicted = model.predict(test_source[i],beam_size=beam_size)
        total_score += bleu_score(predicted,test_target[i])
    
    return total_score/len(test_source)


def problem1():
    n_encoder_blocks,n_decoder_blocks,n_heads = 1,1,1
    train_model(n_encoder_blocks,n_decoder_blocks,n_heads)


def problem2():
    n_encoder_blocks,n_decoder_blocks,n_heads = 1,1,3
    train_model(n_encoder_blocks,n_decoder_blocks,n_heads)

def problem3():
    n_encoder_blocks,n_decoder_blocks,n_heads = 2,2,1
    train_model(n_encoder_blocks,n_decoder_blocks,n_heads)


def problem4():
    n_encoder_blocks,n_decoder_blocks,n_heads = 2,2,3
    train_model(n_encoder_blocks,n_decoder_blocks,n_heads)







if __name__ == "__main__":
    # Loads data from English -> Spanish machine translation task
    problem3()
    problem4()

