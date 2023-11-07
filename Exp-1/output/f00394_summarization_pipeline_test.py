from f00394_summarization_pipeline import *
model_name = "stevhliu/my_awesome_billsum_model"
summarizer = summarization_pipeline(model_name)
text = "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country."
summary = summarizer(text)
assert summary == [{"summary_text": "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country."}]
