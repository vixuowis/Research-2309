from f00552_remove_runaway_bounding_boxes import *
cppe5 = {
    "train": Dataset.from_dict({...}),
    "test": Dataset.from_dict({...}),
    "validation": Dataset.from_dict({...}),
}

remove_idx = [590, 821, 822, 875, 876, 878, 879]

cppe5 = remove_runaway_bounding_boxes(cppe5, remove_idx)

assert len(cppe5["train"]) == 892
assert len(cppe5["test"]) == 100
assert len(cppe5["validation"]) == 100
