from f00158_save_pretrained import *
import tensorflow as tf


# Test case 1
model1 = tf.keras.Sequential()
path1 = "path/to/awesome-name-you-picked1"
save_pretrained(model1, path1)

# Test case 2
model2 = tf.keras.Sequential()
path2 = "path/to/awesome-name-you-picked2"
save_pretrained(model2, path2)

# Test case 3
model3 = tf.keras.Sequential()
path3 = "path/to/awesome-name-you-picked3"
save_pretrained(model3, path3)
