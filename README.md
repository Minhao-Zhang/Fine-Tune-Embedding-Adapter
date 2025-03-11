# Fine Tuning Embedding Model with an Adaptor Layer 

Efficiently fine tune any Sentence Transformer embedding model with your own dataset. 
All you need is your own dataset.

Most of the essential training logic came from [Adam Lucek](https://github.com/ALucek). 
However, I have factored out many of the code to active fine tuning much easier to use. 
In addition, I have improved the logic by pre-computing all embeddings which resulted in drastic incrase in training speed. 
Depending on your embedding model size and data size, your training could be done within minutes after the embedding is done.


## Installation 

Clone this repo, then create a conda enviornment.

```bash
conda create -n embed python=3.13.2
```

Install reruired packages. 

```bash
pip install -r requirements.txt
```

## Fine Tuning

In the `run.py` file, replace train and test with any `pandas.DataFrame` that has two coloumns of `question` and `chunk`.  
You can find any random text to use for the negative. 
I used a book from [Project Gutenberg](https://www.gutenberg.org/). 

That's all you need to do!

## TODO

- [ ] Optimize the choice of hyperparameters

## Acknowledgement

The content of this repo took heavy insipiration from 
1. [Adam Lucek](https://github.com/ALucek/linear-adapter-embedding)'s YouTube video,
2. [Chroma](https://research.trychroma.com/embedding-adapters)'s research report.
