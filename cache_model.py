from transformers import AutoModelForCausalLM, AutoTokenizer

#model_name = "Qwen/Qwen3.5-0.8B"
model_name = "google/gemma-3-1b-it"
local_models_folder = "F:/quantas/models/"
new_model_path = local_models_folder + model_name

# Download and load the model (first run only)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer to a specific local directory
model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)