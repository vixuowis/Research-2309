from f00546_pipeline import *
video_cls = video_classification_pipeline('my_awesome_video_cls_model')
result = video_cls('https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/v_BasketballDunk_g14_c06.avi')
print(result)
