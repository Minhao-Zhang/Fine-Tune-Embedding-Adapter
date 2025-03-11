from data_prep import get_negative_samples
from sentence_transformers import SentenceTransformer
from adapter_trainer import AdapterTrainer
import pandas as pd

train = pd.read_json("data/mc_cq_train.json")
test = pd.read_json("data/mc_cq_test.json")


negative = get_negative_samples("data/pg84.txt")

base_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

trainer = AdapterTrainer(base_model, train, test, negative, device="cuda")

trainer.calculate_baseline()

trainer.train(save_path="models", num_epochs=1000,
              warmup_steps=10, learning_rate=5e-3)
