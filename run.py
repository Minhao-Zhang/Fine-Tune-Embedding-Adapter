from data_prep import get_negative_samples
from sentence_transformers import SentenceTransformer
from adapter_trainer import AdapterTrainer
import pandas as pd
import json


# train = pd.read_json("data/adam_train.json")
# test = pd.read_json("data/adam_test.json")
train = pd.read_parquet("data/new_train.parquet")
test = pd.read_parquet("data/new_test.parquet")
# with open("data/all_chunks.json", "r") as f:
#     all_chunks = list(json.load(f))

# all_chunks = ["search_document: " + c for c in all_chunks]
# train['chunk'] = "search_document: " + train['chunk']
# test['chunk'] = "search_document: " + test['chunk']
# train['qeustion'] = "search_query: " + train['question']
# test['qeustion'] = "search_query: " + test['question']

negative = get_negative_samples("data/pg84.txt")
negative.extend(get_negative_samples("data/pg145.txt"))
negative.extend(get_negative_samples("data/pg174.txt"))
negative.extend(get_negative_samples("data/pg394.txt"))
negative.extend(get_negative_samples("data/pg1342.txt"))
negative.extend(get_negative_samples("data/pg2641.txt"))
negative.extend(get_negative_samples("data/pg2701.txt"))

# all-MiniLM-L12-v2
# all-mpnet-base-v2

base_model = SentenceTransformer(
    "all-MiniLM-L12-v2", trust_remote_code=True, device="cuda")

trainer = AdapterTrainer(base_model, train, test,
                         negative, all_chunks=None, device="cuda")

trainer.calculate_baseline()

trainer.train(
    num_epochs=2000,
    batch_size=256,
    eval_epoch=50,
    save_epoch=200,
    save_path="models"
)
