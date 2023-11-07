from f00217_get_predicted_class_label import *
import torch


# Test case 1
logits1 = torch.tensor([0.2, 0.8, 0.1])
model1 = PreTrainedModel()
assert get_predicted_class_label(logits1, model1) == 'POSITIVE', 'Test Case 1 Failed'

# Test case 2
logits2 = torch.tensor([0.1, 0.3, 0.6])
model2 = PreTrainedModel()
assert get_predicted_class_label(logits2, model2) == 'NEGATIVE', 'Test Case 2 Failed'

# Test case 3
logits3 = torch.tensor([0.4, 0.4, 0.2])
model3 = PreTrainedModel()
assert get_predicted_class_label(logits3, model3) == 'NEUTRAL', 'Test Case 3 Failed'
