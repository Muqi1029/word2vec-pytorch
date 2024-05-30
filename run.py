#!/usr/local/anaconda3/bin/python

import torch
import numpy as np
from dataset.treebank import StanfordSentiment
import argparse
from utils import seed_everything, ProgressMeter, AverageMeter
from word2vec import skipgram
import matplotlib.pyplot as plt
from torch import Tensor
import time
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

import sys
assert sys.version_info.major == 3
assert sys.version_info.minor >= 5

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Word2vec training")
    parser.add_argument("--seed", type=int, default=2024, 
                        help="Random seed for reproducibility (default: 2024)")
    parser.add_argument("--data_dir", required=True, type=str, 
                        help="Directory containing the training data")
    parser.add_argument("--dim_vectors", type=int, default=10, 
                        help="Dimensionality of the word vectors (default: 10)")
    parser.add_argument("--context_size", type=int, default=5, 
                        help="Size of the context window (default: 5)")
    parser.add_argument("--resume", type=str, 
                        help="Path to a checkpoint to resume training from")
    parser.add_argument("--batch_size", type=int, default=64, 
                        help="Batch size for training (default: 64)")
    parser.add_argument("--lr", type=float, default=0.01, 
                        help="Learning rate for training (default: 0.01)")
    parser.add_argument("--num_iterations", type=int, default=10, 
                        help="Number of iterations for training (default: 10)")
    parser.add_argument("--using_negative_sample", action="store_true", default=False, 
                        help="Use negative sampling (default: False)")
    return parser.parse_args(args)


def main(args):
    seed_everything(args.seed)

    # 1. get dataset from the document
    data = StanfordSentiment(path=args.data_dir)

    # 2. initialize the word vectors
    if args.resume:
        pass
    else:
        nWords = len(data.tokens())
        centerWordVectors = (torch.rand(nWords, args.dim_vectors) -
                        0.5) / args.dim_vectors
        centerWordVectors.requires_grad = True
        outsideWordVectors = torch.zeros(nWords, args.dim_vectors, requires_grad=True)
        wordVectors = torch.cat([centerWordVectors, outsideWordVectors], dim=0)

    # 3. train the word vectors
    optimizer, scheduler = get_optimizer(centerWordVectors, outsideWordVectors, args.lr)
    train(data, optimizer, scheduler, skipgram, centerWordVectors, outsideWordVectors, args)

    # 4. save the word vectors
    torch.save(wordVectors, "checkpoint/wordVectors.params")

    # 5. visualize the word vectors
    visualizeWords = [
        "great", "cool", "brilliant", "wonderful", "well", "amazing",
        "worth", "sweet", "enjoyable", "boring", "bad", "dumb",
        "annoying", "female", "male", "queen", "king", "man", "woman", "rain", "snow",
        "hail", "coffee", "tea"]
    visualize(visualizeWords, data.tokens(), wordVectors, nWords)


def train(data, optimizer, scheduler, wordModel, centerWordVectors, outsideWordVectors, args):
    print(f"using negative sampling for word vectors: {args.using_negative_sample}".center(55, "="))
    for iteration in range(args.num_iterations):
        loss_metric = AverageMeter(name="loss", fmt=":f")
        time_metric = AverageMeter(name="time", fmt=":.3f")
        progress = ProgressMeter(args.num_iterations, 
                                 [loss_metric, time_metric],
                                 prefix="Iteration")

        optimizer.zero_grad()
        start_time = time.time()
        for _ in range(args.batch_size):
            # windowSize1 = random.randint(1, args.context_size)
            centerWord, context = data.getRandomContext(args.context_size)

            l = wordModel(centerWord, 
                          context, 
                          centerWordVectors,
                          outsideWordVectors,
                          data.tokens(),
                          data,
                          args.using_negative_sample)
            loss_metric.update(l.item())
            l.backward()
        optimizer.step()
        scheduler.step()
        time_metric.update(time.time() - start_time)
        progress.display(iteration + 1)
            
        
def get_optimizer(centerWordVectors: Tensor, outsideWordVectors: Tensor, lr: float):
    optimizer = torch.optim.SGD([centerWordVectors, outsideWordVectors], lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    return optimizer, scheduler


def visualize(words, tokenizer, wordVectors, nWords):
    wordsIndices = [tokenizer[word] for word in words]
    wordVecs = torch.cat([wordVectors[:nWords], wordVectors[nWords:]], dim=1)
    
    # svd
    temp = (wordVecs - torch.mean(wordVecs, dim=0))
    covariance = 1.0 / len(wordsIndices) * temp.T @ temp
    U, S, V = np.linalg.svd(covariance.detach().numpy())
    coord = temp.detach().numpy() @ U[:, 0:2]
    for i in range(len(wordsIndices)):
        plt.text(coord[i, 0], coord[i, 1], words[i],
                 bbox=dict(facecolor='green', alpha=0.1))
    plt.xlim((np.min(coord[:, 0]), np.max(coord[:, 0])))
    plt.ylim((np.min(coord[:, 1]), np.max(coord[:, 1])))
    plt.savefig("word_vectors.png", dpi=800)
   

if __name__ == '__main__':
    args = parse_args()
    main(args)
