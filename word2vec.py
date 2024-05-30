from typing import Dict, List
import numpy as np
import argparse
import random
import torch.nn.functional as F
import torch
from torch import Tensor


def negativeSamplingLoss(currentCenterWord: str,
                         outsideWords: List[str],
                         centerVectors: Tensor,
                         outsideVectors: Tensor,
                         word2Ind: Dict,
                         data,
                         K: int = 10,
                         ):
    outsideWordsIndices = [word2Ind[outsideWord] for outsideWord in outsideWords]

    centerWordInd = word2Ind[currentCenterWord]
    centerWordVector = centerVectors[centerWordInd]
    loss = 0
    for outsideWordInd in outsideWordsIndices:
        negSampleWordIndices = []
        for _ in range(K):
            idx = data.sampleTokenIdx()
            while idx in outsideWordsIndices:
                idx = data.sampleTokenIdx()
            negSampleWordIndices.append(idx)

        negativeWordVectors = outsideVectors[negSampleWordIndices]

        loss += - F.sigmoid(centerWordVector @ outsideVectors[outsideWordInd]) \
            - torch.sum(torch.log(F.sigmoid(-negativeWordVectors @ centerWordVector)))
    return loss / len(outsideWords)
    
    

def skipgram(currentCenterWord: str,
             outsideWords: List[str],
             centerVectors: Tensor,
             outsideVectors: Tensor,
             word2Ind: Dict,
             data,
             isNegSample: bool = False):
    
    if isNegSample:
        return negativeSamplingLoss(currentCenterWord=currentCenterWord,
                                    outsideWords=outsideWords,
                                    centerVectors=centerVectors,
                                    outsideVectors=outsideVectors,
                                    word2Ind=word2Ind,
                                    data=data)
        
    centerInd = word2Ind[currentCenterWord]
    centerVector = centerVectors[centerInd]
    # print(f"check whether center word vector needs gradient: {centerVector.requires_grad}")

    # Calculate scores (logits) for all outside words
    scores = outsideVectors @ centerVector  # Shape: (vocab_size,)
    probs = F.softmax(scores, dim=-1)

    loss = 0.0
    for outsideWord in outsideWords:
        outsideWordIndex = word2Ind[outsideWord]

        # Create a tensor with the true label index
        true_label = torch.tensor([outsideWordIndex], dtype=torch.long)
        
        loss += -torch.log(probs[true_label])

        # loss += F.cross_entropy(scores.view(1, -1), true_label)

    return loss / len(outsideWords)
