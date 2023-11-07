from f00334_run_fill_mask_pipeline import *
text = "The Milky Way is a [MASK] galaxy."
model_name = "stevhliu/my_awesome_eli5_mlm_model"
top_k = 3

predictions = run_fill_mask_pipeline(text, model_name, top_k)
print(predictions)
