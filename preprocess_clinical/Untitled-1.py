# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense,Dropout,MaxPooling1D, Flatten,BatchNormalization, GaussianNoise,Conv1D
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils import compute_class_weight
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential, save_model, load_model

# %%
#this was created in general/diagnosis_making notebook
diag = pd.read_csv("../general/ground_truth.csv").drop("Unnamed: 0", axis=1)

# %% [markdown]
# Below we are combining several clinical datasets.

# %%
demo = pd.read_csv("../data/PTDEMOG.csv")

# %%
neuro = pd.read_csv("../data/NEUROEXM.csv")

# %%
neuro.columns

# %%
clinical = pd.read_csv("../data/ADSP_PHC_COGN.csv") #.rename(columns={"PHASE":"PHASE"})

# %%
clinical.head()

# demo.head()

# %%
diag["Subject"].value_counts()

# %%
comb = pd.read_csv("../data/DXSUM_PDXCONV_ADNIALL.csv")[["RID", "PTID" , "PHASE"]]

# %%
m = comb.merge(demo, on = ["RID", "PHASE"]).merge(neuro,on = ["RID", "PHASE"]).merge(clinical,on = ["RID", "PHASE"]).drop_duplicates()

# %%
m.columns = [c[:-2] if str(c).endswith(('_x','_y')) else c for c in m.columns]

m = m.loc[:,~m.columns.duplicated()]

# %%
diag = diag.rename(columns = {"Subject": "PTID"})

# %%
m = m.merge(diag, on = ["PTID", "PHASE"])

# %%
m["PTID"].value_counts()

# %%
t = m

# %%
t = t.drop(["ID",  "SITEID", "VISCODE", "VISCODE2", "USERDATE", "USERDATE2",
            "update_stamp",  "PTSOURCE","DX"], axis=1) 

# %%
t.columns

# %%
t = t.fillna(-4)
t = t.replace("-4", -4)
cols_to_delete = t.columns[(t == -4).sum()/len(t) > .70]
t.drop(cols_to_delete, axis = 1, inplace = True)

# %%
len(t.columns)
print(t.columns)
print(cols_to_delete)

# %%
# t["PTWORK"] = t["PTWORK"].str.lower().str.replace("housewife", "homemaker").str.replace("rn", "nurse").str.replace("bookeeper",
                                                                                                                # "bookkeeper").str.replace("cpa", "accounting")

# %%
# t["PTWORK"] = t["PTWORK"].fillna("-4").astype(str)

# %%
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*teach.*$)', 'education')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*bookkeep.*$)', 'bookkeeper')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*wife.*$)', 'homemaker')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*educat.*$)', 'education')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*engineer.*$)', 'engineer')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*eingineering.*$)', 'engineer') 
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*computer programmer.*$)', 'engineer') 
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*nurs.*$)', 'nurse')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*manage.*$)', 'managment')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*therapist.*$)', 'therapist')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*sales.*$)', 'sales')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*admin.*$)', 'admin')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*account.*$)', 'accounting')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*real.*$)', 'real estate')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*secretary.*$)', 'secretary')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*professor.*$)', 'professor')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*chem.*$)', 'chemist')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*business.*$)', 'business')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*writ.*$)', 'writing')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*psych.*$)', 'psychology')
# t['PTWORK'] = t['PTWORK'].str.replace(r'(^.*analys.*$)', 'analyst')

# %%
# cond = t['PTWORK'].value_counts()
# threshold = 10
# t['PTWORK'] = np.where(t['PTWORK'].isin(cond.index[cond >= threshold ]), t['PTWORK'], 'other')

# %%
categorical = ['PTGENDER',
 'PTHOME',
 'PTMARRY',
 'PTEDUCAT',
 'PTPLANG',
 'NXVISUAL',
 'PTNOTRT',
 'NXTREMOR',
 'NXAUDITO',
 'PTHAND']

# %%
quant = ['PTDOBYY',
 'PHC_MEM',
 'PHC_EXF',
 'PTRACCAT',
 'AGE',
 'PTADDX',
 'PTETHCAT',
 'PTCOGBEG',
 'PHC_VSP',
 'PHC_LAN']

# %%
text = ["PTWORK", "CMMED", "PTDOB", "VISDATE"]

# %%
cols_left = list(set(t.columns) - set(categorical) - set(text)  - set(["label", "Group","GROUP", "PHASE", "RID", "PTID"]))
t[cols_left]

# %%
for col in cols_left:
    if len(t[col].value_counts()) < 10:
        print(col)
        categorical.append(col)

# %%
to_del = ["PTRTYR", "EXAMDATE", "SUBJECT_KEY"]
t = t.drop(to_del, axis=1)

# %%
quant = list(set(cols_left) - set(categorical) - set(text)  -set(to_del) - set(["label", "Group","GROUP", "PHASE", "RID", "PTID"]))
t[quant]

# %%
cols_left = list(set(cols_left) - set(categorical) - set(text) - set(quant) - set(to_del))

# %%
#after reviewing the meaning of each column, these are the final ones
l = ['RID', 'PTID', 'Group', 'PHASE', 'PTGENDER', 'PTDOBYY', 'PTHAND',
       'PTMARRY', 'PTEDUCAT', 'PTNOTRT', 'PTHOME', 'PTTLANG',
       'PTPLANG', 'PTCOGBEG', 'PTETHCAT', 'PTRACCAT', 'NXVISUAL',
       'NXAUDITO', 'NXTREMOR', 'NXCONSCI', 'NXNERVE', 'NXMOTOR', 'NXFINGER',
       'NXHEEL', 'NXSENSOR', 'NXTENDON', 'NXPLANTA', 'NXGAIT', 
       'NXABNORM',  'PHC_MEM', 'PHC_EXF', 'PHC_LAN', 'PHC_VSP']

# %%
t[l]

# %%
dfs = []

# %%
for col in categorical:
    dfs.append(pd.get_dummies(t[col], prefix = col, dtype=float))

# %%
cat = pd.concat(dfs, axis=1)

# %%
t[quant]

# %%
cat

# %%
t[["PTID","RID", "PHASE", "Group"]]

# %%
c = pd.concat([t[["PTID", "RID", "PHASE", "Group"]].reset_index(), cat.reset_index(), t[quant].reset_index()], axis=1).drop("index", axis=1) #tex

# %%
c

# %%
#removing repeating subjects, taking the most recent diagnosis
c = c.groupby('PTID', 
                  group_keys=False).apply(lambda x: x.loc[x["Group"].astype(int).idxmax()]).drop("PTID", axis = 1).reset_index(inplace=False)

# %%
c.to_csv("clinical.csv")

# %%
#reading in the overlap test set
# ts = pd.read_csv("overlap_test_set.csv").rename(columns={"subject": "PTID"})

# #removing ids from the overlap test set
# c = c[~c["PTID"].isin(list(ts["PTID"].values))]

# %%
cols = list(set(c.columns) - set(["PTID","RID","subject", "ID","GROUP", "Group", "label", "PHASE", "SITEID", "VISCODE", "VISCODE2", "USERDATE", "USERDATE2", "update_stamp", "DX_x","DX_y", "Unnamed: 0"]))
X = c[cols].values 
y = c["Group"].astype(int).values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train[:1])

# %%
import pickle

print("X_train shape: ", X_train.shape, "y_train shape: ", y_train.shape, "X_test shape: ", X_test.shape, "y_test shape: ", y_test.shape)

with open('X_train_c.pkl', 'wb') as f:
    pickle.dump(X_train, f)

with open('X_test_c.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('y_train_c.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('y_test_c.pkl', 'wb') as f:
    pickle.dump(y_test, f)

# X_train.to_pickle("X_train_c.pkl")
# y_train.to_pickle("y_train_c.pkl")

# X_test.to_pickle("X_test_c.pkl")
# y_test.to_pickle("y_test_c.pkl")

# %%



