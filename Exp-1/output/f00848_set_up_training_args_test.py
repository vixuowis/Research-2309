from f00848_set_up_training_args import *
def test_set_up_training_args():
    args = set_up_training_args()
    assert args["output_dir"] == "tmp"
    assert args["evaluation_strategy"] == "steps"
    assert args["num_train_epochs"] == 1
    assert args["log_level"] == "error"
    assert args["report_to"] == "none"

