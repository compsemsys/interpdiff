from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en")
dataset_name = "wikimedia/wikipedia/20231101.en"
local_models_folder = "F:/quantas/datasets/"
dataset_path = local_models_folder + dataset_name

ds.save_to_disk(dataset_path)
