import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler

df = pd.read_csv(r"diabetes.csv")
x=df[df.columns[:-1]].values
y=df[df.columns[-1]].values
scaler = StandardScaler()
x=scaler.fit_transform(x)
over = RandomOverSampler()
x,y = over.fit_resample(x,y)

x_train, x_temp, y_train, y_temp = train_test_split(x,y,test_size=0.4,random_state=0)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp,y_temp,test_size=0.5,random_state=0)

mod_path = 'diabetes_model.keras'
if os.path.exists(mod_path):
    print("Loading saved model")
    yuga = tf.keras.models.load_model(mod_path)
else:
    yuga = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    yuga.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss="binary_crossentropy",metrics=["accuracy"])
    yuga.fit(x_train,y_train,batch_size=16,epochs=60,validation_data=(x_valid,y_valid))
    yuga.save(mod_path)

yp = yuga.predict(x_test)
yp = (yp > 0.5).astype(int)
acc = accuracy_score(y_test,yp)
print(f"Model accuracy : {acc:.2f}")

def db(a):
    l=list(a.keys())
    l=[a[i] for i in l]
    ui = np.array([l])
    ui = scaler.transform(ui)
    res=yuga.predict(ui)[0][0]
    res = (res > 0.5).astype(int)
    if res==0:
       return("\nYou dont have diabetes\n")
    else:
       return("\nYou have diabetes\n")

gl = float(input("\nEnter your glucose level : "))
bp = float(input("\nEnter your Blood Pressure : "))
ins = float(input("\nEnter your Insulin : "))
bmi = float(input("\nEnter your BMI : "))
age = float(input("\nEnter your age : "))
print(db({"glucose": gl,"BP": bp,"insulin": ins,"BMI": bmi,"age": age}))