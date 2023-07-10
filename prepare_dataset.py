from datasets import load_dataset
from tqdm import tqdm
import json

dataset_name = "wikitext"
dataset_subset_name = "wikitext-103-raw-v1"

ds = load_dataset(dataset_name, dataset_subset_name)
filtered_ds = ds.filter(lambda x: len(x["text"]) > 100)


for split_name in filtered_ds.keys():
    with open(f"./wikitext_{split_name}.jsonl", "w") as ofh:
        for item in tqdm(filtered_ds[split_name], desc=f"writing {split_name} split"):
            json.dump(item, ofh)
            ofh.write("\n")
