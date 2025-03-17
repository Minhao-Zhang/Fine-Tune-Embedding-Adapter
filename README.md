# Efficiently Fine Tune Adapter Layer for Embedding Model

## Theory

Just like you can use LoRA to train an adapter for all the projection matrices for an LLM, you can also do the same for the query embedding as well. 

The basic idea is that we keep the embedding model the same, the documnet embedding the same, but we add a layer to the embedding whenever we perform an query. 
In this case, all your existing embedding's in the vector database does not need to be re-calculated. 
The only thing different is that you feed the result of your query embedding into the adapter layer which will produce a "better" embedding for your domain. 

You can learn more on [Chroma's research report](https://research.trychroma.com/embedding-adapters) and [Adam Lucek's YouTube video](https://www.youtube.com/watch?v=hztWQcoUbt0&feature=youtu.be) where I learned a lot from them.

## Features

- Clearly defined and reusable training modules
- Arbitrary support for [Sentence Transformer](https://sbert.net/) models
- Pre-compute all embeddings to make the training extremely fast (You can see results in seconds after all embeddings are done).
- Custom evaluation metrics (Hit@k & MRR@k built in)
- Support of different adapters (Linear and MLP built in)

## Installation 

Clone this repo, then create a conda enviornment.

```bash
conda create -n embed python=3.13.2
conda activate embed
```

Install reruired packages.

```bash
pip install -r requirements.txt
```

## Fine Tuning

In the `run.py` file, replace train and test with any `pandas.DataFrame` that has two coloumns of `question` and `chunk`.  
You can find any random text to use for the negative. 
I used a book from [Project Gutenberg](https://www.gutenberg.org/). 

```python
from data_prep import get_negative_samples
from sentence_transformers import SentenceTransformer
from adapter_trainer import AdapterTrainer
import pandas as pd

train = pd.read_json("data/train.json")
test = pd.read_json("data/test.json")
negative = get_negative_samples("data/pg84.txt")

base_model = SentenceTransformer("all-MiniLM-L12-v2", device="cuda")

trainer = AdapterTrainer(base_model, train, test, negative, device="cuda")
trainer.calculate_baseline()
trainer.train(
    num_epochs=1000,
    batch_size=256,
    eval_epoch=200,
)
```

That's all you need to do!

## Picking model and adapter

With my limited testing, for models like `all-MiniLM-L6-v2` with dimension of 384, it works better if you use `MLPAdapter` with a large hidden dimension of 1024 or 2048.
For models like `all-mpnet-base-v2` with dimension of 768, a simple `LinearAdapter` will work pretty well.

## Results 

Note, the result of this approach varies drastically with your dataset and your questions. 
I tried this on two of my own RAG systems, they showed significiant improvement in one and obvious performance degradation in the other.
Please experiement with before actually apply it.
But you can always undo this by removing the adapter layer. 

## Acknowledgement

The content of this repo took heavy insipiration from

1. [Adam Lucek](https://github.com/ALucek/linear-adapter-embedding)'s YouTube video,
2. [Chroma](https://research.trychroma.com/embedding-adapters)'s research report.
