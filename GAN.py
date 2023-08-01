"""
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
"""

import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt

# See TensorFlow version
print(tf.__version__)

"""
X_train=[]
Y_train=[]

X_test=[]
Y_test=[]

i=os.listdir("./Original/class2")
n=0
for target_file in i:
    image=("./Original/class2/"+target_file)
    temp_img=load_img(image)
    temp_img_array=img_to_array(temp_img)
    print(temp_img_array.shape)
    X_train.append(temp_img_array)
    n=n+1

np.savez("./gan/gan.npz",x_train=X_train,y_train=Y_train,x_test=X_test,y_test=Y_test)
"""