from data_prep import get_negative_samples
from sentence_transformers import SentenceTransformer
from adapter_trainer import AdapterTrainer
import pandas as pd


train = pd.read_json("data/new_train.json")
test = pd.read_json("data/new_test.json")

negative = get_negative_samples("data/pg84.txt")
negative.extend(get_negative_samples("data/pg145.txt"))
negative.extend(get_negative_samples("data/pg174.txt"))
negative.extend(get_negative_samples("data/pg394.txt"))
negative.extend(get_negative_samples("data/pg1342.txt"))
negative.extend(get_negative_samples("data/pg2641.txt"))
negative.extend(get_negative_samples("data/pg2701.txt"))


base_model = SentenceTransformer("all-MiniLM-L12-v2", device="cuda")

trainer = AdapterTrainer(base_model, train, test, negative, device="cuda")

trainer.calculate_baseline()

trainer.train(
    num_epochs=1000,
    batch_size=256,
    eval_epoch=200,
)
