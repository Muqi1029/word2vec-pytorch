import pytest
import torch
from dataset.treebank import StanfordSentiment
from word2vec import skipgram


class TestSkipGram:
    @pytest.fixture(autouse=True)
    def pre_run_test(self):
        self.word_dim = 10
        self.data = StanfordSentiment(path="stanfordSentimentTreebank")
        self.word2Ind = self.data.tokens()
        self.nWords = len(self.word2Ind)
        self.centerVectors = (torch.randn(self.nWords, self.word_dim) - 0.5) / self.word_dim
        self.centerVectors.requires_grad_()
        self.outsideVectors = torch.zeros(self.nWords, self.word_dim, requires_grad=True)
        self.currentCenterWord, self.outsideWords = self.data.getRandomContext()
        
    def test_naive_skip_gram(self):
        loss = skipgram(self.currentCenterWord,
                        self.outsideWords,
                        self.centerVectors,
                        self.outsideVectors,
                        self.word2Ind,
                        self.data)
        print(f"Loss: {type(loss)} {loss.item()}")


    def test_negative_sample_skip_gram(self):
        loss = skipgram(self.currentCenterWord,
                        self.outsideWords,
                        self.centerVectors,
                        self.outsideVectors,
                        self.word2Ind, 
                        self.data,
                        isNegSample=True)
        print(f"Loss: {type(loss)} {loss.item()}")
    
    
    def test_check_grad(self):
        dim_vectors = 5  # Example dimension size
        centerWordVectors = (torch.rand(self.nWords, dim_vectors) - 0.5) / dim_vectors
        centerWordVectors.requires_grad_()
        outsideWordVectors = torch.zeros(self.nWords, dim_vectors, requires_grad=True)
        loss = skipgram(self.currentCenterWord,
                        self.outsideWords, 
                        centerVectors=centerWordVectors,
                        outsideVectors=outsideWordVectors,
                        word2Ind=self.word2Ind,
                        data=self.data)
        print(f"loss {loss.item()}")
        print(f"before backward".center(50, "="))
        print(centerWordVectors.grad)
        print(outsideWordVectors.grad)
        loss.backward()
        print(f"after backward".center(50, "="))
        print(centerWordVectors.grad)
        print(outsideWordVectors.grad)

        
                        
        