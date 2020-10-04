# Homework 4
# David Armando SANCHEZ
# October 3, 2020

import numpy as np
import pandas as pd

data = np.genfromtxt("train_heating_2008.txt", dtype=None)  # We import the data with numpy
data = pd.DataFrame(data)   # Once imported, we use pandas to build our data frame
pd.DataFrame.head(data)

N = np.shape(data)[0]
data.columns = ["fix1", "fix2", "fix3", "fix4", "fix5",
                "var1", "var2", "var3", "var4", "var5",
                "income", "choice"]

x1 = data[["fix1", "var1"]]
x2 = data[["fix2", "var2"]]
x3 = data[["fix3", "var3"]]
x4 = data[["fix4", "var4"]]
x5 = data[["fix5", "var5"]]

x1 = x1.assign(c1=pd.DataFrame(np.repeat(0, N)),
               c2=pd.DataFrame(np.repeat(0, N)),
               c3=pd.DataFrame(np.repeat(0, N)),
               c4=pd.DataFrame(np.repeat(0, N)))
x2 = x2.assign(c1=pd.DataFrame(np.repeat(1, N)),
               c2=pd.DataFrame(np.repeat(0, N)),
               c3=pd.DataFrame(np.repeat(0, N)),
               c4=pd.DataFrame(np.repeat(0, N)))
x3 = x3.assign(c1=pd.DataFrame(np.repeat(0, N)),
               c2=pd.DataFrame(np.repeat(1, N)),
               c3=pd.DataFrame(np.repeat(0, N)),
               c4=pd.DataFrame(np.repeat(0, N)))
x4 = x4.assign(c1=pd.DataFrame(np.repeat(0, N)),
               c2=pd.DataFrame(np.repeat(0, N)),
               c3=pd.DataFrame(np.repeat(1, N)),
               c4=pd.DataFrame(np.repeat(0, N)))
x5 = x5.assign(c1=pd.DataFrame(np.repeat(0, N)),
               c2=pd.DataFrame(np.repeat(0, N)),
               c3=pd.DataFrame(np.repeat(0, N)),
               c4=pd.DataFrame(np.repeat(1, N)))
# Estimation of means
m1 = np.mean(x1, axis=0)
m2 = np.mean(x2, axis=0)
m3 = np.mean(x3, axis=0)
m4 = np.mean(x4, axis=0)
m5 = np.mean(x5, axis=0)

m1 = m1.tolist()
m2 = m2.tolist()
m3 = m3.tolist()
m4 = m4.tolist()
m5 = m5.tolist()

means = pd.DataFrame({"x1": m1, "x2": m2, "x3": m3, "x4": m4, "x5": m5})
print(means)

# Estimation of standard errors
std1 = np.std(x1, axis=0)
std2 = np.std(x2, axis=0)
std3 = np.std(x3, axis=0)
std4 = np.std(x4, axis=0)
std5 = np.std(x5, axis=0)

std1 = std1.tolist()
std2 = std2.tolist()
std3 = std3.tolist()
std4 = std4.tolist()
std5 = std5.tolist()

Standard_dev = pd.DataFrame({"x1": std1, "x2": std2, "x3": std3, "x4": std4, "x5": std5})
print(Standard_dev)

# Utility

b = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4])

e1 = np.exp(np.dot(x1.values, b))
e2 = np.exp(np.dot(x2.values, b))
e3 = np.exp(np.dot(x3.values, b))
e4 = np.exp(np.dot(x4.values, b))
e5 = np.exp(np.dot(x4.values, b))

exp = pd.DataFrame({"e1": e1, "e2": e2, "e3": e3, "e4": e4, "e5": e5})

res = pd.DataFrame({"sum": pd.DataFrame.sum(exp, 1)})
res["l1"] = np.log(exp["e1"]/res["sum"])
res["l2"] = np.log(exp["e2"]/res["sum"])
res["l3"] = np.log(exp["e3"]/res["sum"])
res["l4"] = np.log(exp["e4"]/res["sum"])
res["l5"] = np.log(exp["e5"]/res["sum"])
res["choice"] = data["choice"]

