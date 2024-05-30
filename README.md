# A Simple WordVector Implementation in PyTorch

## Project Structure
```
.
├── checkpoint                  # Directory to store model checkpoints
│   └── wordVectors.params     
├── dataset                     # Directory for dataset-related files
│   └── treebank.py            # Code file for dataset processing
├── pytest.ini                  # Configuration file for pytest
├── README.md                  
├── requirements.txt           
├── run.py                      # Main script for running the project
├── scripts                     # Directory for shell scripts
│   └── run.sh                 # Sample shell script for running the project
├── stanfordSentimentTreebank   # Directory containing dataset files
│   ├── datasetSentences.txt   
│   ├── datasetSplit.txt       
│   ├── dictionary.txt         
│   ├── original_rt_snippets.txt 
│   ├── README.txt            
│   ├── sentiment_labels.txt  
│   ├── SOStr.txt             
│   └── STree.txt             
├── tests                       # Directory for test files
│   ├── test_dataset.py        
│   ├── test_run.py            
│   └── test_skipgram.py       
├── utils.py                    # Utility functions file
├── word2vec.py                 # Main script for word2vec implementation
└── word_vectors.png            # Image file depicting word vectors
```

## Usage
```sh
./scripts/run.sh <num_iterations>
```

## Reference:
- - [CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/): Stanford course taught by Christopher Manning and Richard Socher. The project structure and organization draw inspiration from the assignments and materials covered in this course.
