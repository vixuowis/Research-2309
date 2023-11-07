from f00837_masked import *
text = "Hugging Face is a community-based open-source <mask> for machine learning."
preds = masked(text)
print(preds)
