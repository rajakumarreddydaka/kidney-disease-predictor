

# Importing necessary libraries used for data cleaning, and data visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns

# Ignoring ununnecessary warnings
import warnings
warnings.filterwarnings("ignore")

# Importing library to split the data into training part and testing part.
from sklearn.model_selection import train_test_split

# Chi Square
from sklearn.feature_selection import chi2
import scipy.stats as stats

# Importing library to process the data (Normalize the data)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Importing Models (used for making prediction)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC                            # Support vector machine model
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore

# Importing metrics used for evaluation of our models
from sklearn import metrics
from sklearn.metrics import classification_report

# Hyperparameter tuner and Cross Validation
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# RandomOverSampler to handle imbalanced data
from imblearn.over_sampling import RandomOverSampler

sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
pd.set_option('display.max_columns',29)

"""# Data Collection"""

df = pd.read_csv("kidney.csv")

df.head()

df.tail()

 

# Exploratory Data Analysis (EDA)

# Checking the number of rows and columns in our dataset
df.shape

"""- Dataset contains 400 rows and 26 columns"""

# Getting more information of our dataset
df.info()



df.isnull().sum()


# Getting some statistical information of our data
df.describe()



# Distribution of our target variable i.e. "classification" column
df["classification"].value_counts()

for i in df.drop("id",axis=1).columns:
    print('Unique Values in "{}":\n'.format(i),df[i].unique(), "\n\n")



df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']] = df[['pcv', 'wc', 'rc', 'dm', 'cad', 'classification']].replace(to_replace={'\t8400':'8400', '\t6200':'6200', '\t43':'43', '\t?':np.nan, '\tyes':'yes', '\tno':'no', 'ckd\t':'ckd', ' yes':'yes'})

"""- Let's check still if there any "\t" in our data"""

for i in df.drop("id",axis=1).columns:
    print('Unique Values in "{}":\n'.format(i),df[i].unique(), "\n\n")



style.use('seaborn-darkgrid')

d = ((df.isnull().sum()/df.shape[0])).sort_values(ascending=False)
# Here we are plotting null values in range of 0-1. It means y axis range is 0-1.
# If bar graph show 0.5 null values that means there are 50% null values in that particular column.
# Hence we are dividing number of null values of each column with total number of rows i.e. 400 (or df.shape[0])

d.plot(kind = 'bar',
       color = sns.cubehelix_palette(start=2,
                                    rot=0.15,
                                    dark=0.15,
                                    light=0.95,
                                    reverse=True,
                                    n_colors=24),
        figsize=(20,10))
plt.title("\nProportions of Missing Values:\n",fontsize=40)
plt.show()



sns.distplot(df.age)



df["age"] = df["age"].replace(np.NaN, df["age"].median())

"""### Blood Pressure (bp)"""

df.bp.unique()

df.bp.mode()[0]

df.bp = df.bp.replace(np.NaN, df.bp.mode()[0])

"""### Specific Gravity (sg)"""

df.sg.unique()

df.sg.mode()[0]

df.sg = df.sg.replace(np.NaN, df.sg.mode()[0])

"""### Aluminium (al)"""

df.al.unique()

df.al.mode()[0]

df.al = df.al.replace(np.NaN, df.al.mode()[0])

"""### Sugar (su)"""

df.su.unique()

df.su.mode()[0]

df.su = df.su.replace(np.NaN, df.su.mode()[0])

"""### Red blood cell (rbc)"""

df.su.unique()

df.rbc.mode()[0]

df.rbc = df.rbc.replace(np.NaN, df.rbc.mode()[0])

"""### pc"""

df.pc.unique()

df.pc.mode()[0]

df.pc = df.pc.replace(np.NaN, df.pc.mode()[0])

"""### pcc"""

df.pcc.unique()

df.pcc.mode()[0]

df.pcc = df.pcc.replace(np.NaN, df.pcc.mode()[0])

"""### ba"""

df.ba.unique()

df.ba.mode()[0]

df.ba = df.ba.replace(np.NaN, df.ba.mode()[0])

"""### bgr"""

sns.distplot(df.bgr)

"""- Seems positive skewed so we will replace nan with median"""

df.bgr.median()

df.bgr = df.bgr.replace(np.NaN, df.bgr.median())

"""### bu"""

sns.distplot(df.bu)

"""- Seems positive skewed so we will replace nan with median"""

df.bu.median()

df.bu = df.bu.replace(np.NaN, df.bu.median())

"""### sc"""

sns.distplot(df.sc)

"""- Seems positive skewed so we will replace nan with median"""

df.sc.median()

df.sc = df.sc.replace(np.NaN, df.sc.median())

"""### sod"""

sns.distplot(df.sod)

"""- Seems negative skewed so we will replace nan with median"""

df.sod.median()

df.sod = df.sod.replace(np.NaN, df.sod.median())

"""### pot"""

sns.distplot(df.pot)

"""- Seems positive skewed so we will replace nan with median"""

df.pot.median()

df.pot = df.pot.replace(np.NaN, df.pot.median())

"""### hemo"""

sns.distplot(df.hemo)

df.hemo.skew(skipna = True)

"""- Seems little negative skewed so we will replace nan with median"""

df.hemo.median()

df.hemo = df.hemo.replace(np.NaN, df.hemo.median())

"""### pcv"""

sns.distplot(df.pcv)

df.pcv.skew(skipna = True)

"""- Seems little negative skewed so we will replace nan with median"""

df.pcv.median()

df.pcv = df.pcv.replace(np.NaN, df.pcv.median())

"""### wc"""

sns.distplot(df.wc)

"""Seems positive skewed so we will replace nan with median"""

df.wc.median()

df.wc = df.wc.replace(np.NaN, df.wc.median())

"""### rc"""

sns.distplot(df.rc)

df.rc.skew(skipna = True)

"""- Seems little negative skewed so we will replace nan with median"""

df.rc.median()

df.rc = df.rc.replace(np.NaN, df.rc.median())

"""### htn"""

df.htn.unique()

df.htn.mode()

df.htn = df.htn.replace(np.NaN, df.htn.mode()[0])

"""### dm"""

df.dm.mode()

df.dm = df.dm.replace(np.NaN, df.dm.mode()[0])

"""### cad"""

df.cad.unique()

df.cad.mode()

df.cad = df.cad.replace(np.NaN, df.cad.mode()[0])

"""### appet"""

df.appet.unique()

df.appet.mode()

df.appet = df.appet.replace(np.NaN, df.appet.mode()[0])

"""### pe"""

df.pe.unique()

df.pe.mode()

df.pe = df.pe.replace(np.NaN, df.pe.mode()[0])

"""### ane"""

df.ane.unique()

df.ane.mode()

df.ane = df.ane.replace(np.NaN, df.ane.mode()[0])

df.isnull().sum()



df.dtypes

for i in df.columns:
    print('Unique Values in "{}":\n'.format(i), df[i].unique(), "\n-----------------------------------------------------\n")



df['rc'] = df['rc'].astype('float64')
df[['pcv', 'wc', 'age']] = df[['pcv', 'wc', 'age']].astype('int64')
df.dtypes

display(df['pcv'].unique())
display(df['wc'].unique())
display(df['rc'].unique())



df.drop('id',axis=1,inplace=True)
df.head()



sns.countplot(x = "classification", data = df)

pp = sns.pairplot(df[["bp","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","classification"]], hue = "classification", height=1.8, aspect=1.8, plot_kws=dict(edgecolor="k", linewidth=0.5), diag_kind="kde", diag_kws=dict(shade=True))
fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)
t = fig.suptitle('Chronic Kidney Disease', fontsize=30)



sns.set(font_scale=0.45)
plt.title('Chronic Kidney Disease Attributes Correlation')
cmap = sns.diverging_palette(260, 10, as_cmap=True)
sns.heatmap(df[["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc"]].corr("spearman"), vmax=1.2, annot=True, square='square', cmap=cmap, fmt = '.0%', linewidths=2)

# With the following function we can select highly correlated features
# It will remove the first feature that is correlated with anything other feature

def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr("spearman")
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(df, 0.85)
corr_features

sns.scatterplot(x="pcv", y="hemo", data=df)



df.head()

col = ['rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane']
encoder = LabelEncoder()
for col in col:
    df[col] = encoder.fit_transform(df[col])

df[['appet', 'classification']] = df[['appet', 'classification']].replace(to_replace={'good':'1', 'ckd':'1', 'notckd':'0', 'poor':'0'})

df.head(2)

df.dtypes

df[['classification', 'appet']] = df[['classification', 'appet']].astype('int64')



df_anova = df[["age","bp","bgr","bu","sc","sod","pot","hemo","pcv","wc","rc","classification"]]
grps = pd.unique(df_anova.classification.values)
grps

for i in range(len(df_anova.columns)-1):
    
    d_data = {grp:df_anova[df_anova.columns[i]][df_anova.classification == grp] for grp in grps}

    F, p = stats.f_oneway(d_data[0], d_data[1])
    print("P_Value of {} and Classification".format(df_anova.columns[i]), p)

    if p < 0.05:
        print("There is relation between {} and Classification \n".format(df_anova.columns[i]))
    else:
        print("There is no relation between {} and Classification \n".format(df_anova.columns[i]))



x = np.array(df.pot)
y = np.array(df.classification)
_, p = stats.pointbiserialr(x, y)
print(p)

if p < 0.05:
    print("There is relation between Potassium and Classification \n")
else:
    print("There is no relation between Potassium and Classification \n")


df.drop(["pcv","pot"], axis=1, inplace=True)
display(df.head())
df.shape



X = df.drop("classification", axis=1)
y = df["classification"]
display(X)
display(y)



f_p_values=chi2(X[['sg', 'al', 'su', 'rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane', 'appet']],y)

p_values = pd.Series(f_p_values[1])
p_values.index = ['sg', 'al', 'su', 'rbc', 'pcc', 'pc', 'ba', 'htn', 'dm', 'cad', 'pe', 'ane', 'appet']
p_values.sort_values(ascending=False)

# Null Hypothesis: The null hypothesis states that there is no relationship between the two variables
cnt = 0
for i in p_values:
    if i > 0.05:
        print("There is no relationship", p_values.index[cnt], i)
    else:
        print("There is relationship", p_values.index[cnt], i)
    
    cnt += 1

p_values.index



df.drop("sg", axis=1, inplace=True)
display(df.head(2))
df.shape

"""### Dropping constant feature"""

from sklearn.feature_selection import VarianceThreshold

var_thres = VarianceThreshold(threshold=0)
var_thres.fit(df)

var_thres.get_support()

print(df.columns[var_thres.get_support()])


constant_columns = [column for column in df.columns
                    if column not in df.columns[var_thres.get_support()]]
print(constant_columns)
print(len(constant_columns))
print("Shape: ", df.shape)



X = df.drop("classification", axis=1)
y = df["classification"]

"""## Standardization of the data"""

scaler = StandardScaler()
features = scaler.fit_transform(X)
features



X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)

"""## Balancing Data"""

len(y_train[y_train==1]), len(y_train[y_train==0]), y_train.shape



from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()

X_train_down,y_train_down = rus.fit_resample(X_train, y_train)

print(len(y_train_down[y_train_down==0]), len(y_train_down[y_train_down==1]))
print(len(X_train_down))



os =  RandomOverSampler(sampling_strategy=1)

X_train, y_train = os.fit_resample(X_train, y_train)

print(len(y_train[y_train==0]), len(y_train[y_train==1]))
print(len(X_train))

"""# Model Building

## Logistic Regression
"""

def lr_grid_search(X, y):
    model = LogisticRegression()
    
    # Create a dictionary of all values we want to test
    solvers = ['newton-cg', 'lbfgs', 'liblinear']
    penalty = ['l2']
    c_values = [100, 10, 1.0, 0.1, 0.01]
    
    # define grid search
    param_grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
    grid_result = grid_search.fit(X, y)
    
    return grid_result.best_params_

lr_grid_search(X_train, y_train)

"""### Over sample Logistic"""

lr = LogisticRegression(C=1, penalty='l2', solver='newton-cg')
lr.fit(X_train,y_train)

y_pred_lr = lr.predict(X_test)

print(metrics.classification_report(y_test, y_pred_lr))

lr_score = lr.score(X_train,y_train)
print(lr_score)

lr_score = lr.score(X_test,y_test)
print(lr_score)



lr_tacc = lr.score(X_test,y_test)
lr_train_acc = lr.score(X_train, y_train)

"""#### Confusion matrix of Logistic Regression Model"""

cm = metrics.confusion_matrix(y_test, y_pred_lr, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

"""#### AUC of Logistic Regression Model"""

y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

lr_auc = auc
lr_auc

"""# Under Sample Logistic"""

lr_grid_search(X_train_down, y_train_down)

lr = LogisticRegression(C=10, penalty='l2', solver='newton-cg')
lr.fit(X_train_down,y_train_down)

y_pred_lr = lr.predict(X_test)

print(metrics.classification_report(y_test, y_pred_lr))

lr_score = lr.score(X_train_down,y_train_down)
print(lr_score)

lr_score = lr.score(X_test,y_test)
print(lr_score)

cm = metrics.confusion_matrix(y_test, y_pred_lr, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

lr_tacc_down = lr.score(X_test,y_test)
lr_train_acc_down = lr.score(X_train_down, y_train_down)
lr_auc_down = auc
lr_auc_down

"""## Decision Tree Classifier"""

def dtree_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(2, 15)}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # decision tree model
    dtree = DecisionTreeClassifier()
    
    #use gridsearch to test all values
    dtree_gscv = GridSearchCV(dtree, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    #fit model to data
    dtree_gscv.fit(X, y)
    
    return dtree_gscv.best_params_

dtree_grid_search(X_train, y_train)

"""### Over Sample Decision Tree"""

dTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11)
dTree.fit(X_train, y_train)

print(dTree.score(X_train,y_train))
print(dTree.score(X_test,y_test))

y_pred_dtree = dTree.predict(X_test)

print(metrics.classification_report(y_test, y_pred_dtree))

dt_tacc = dTree.score(X_test,y_test)
dt_train_acc = dTree.score(X_train, y_train)

"""#### Confusion Matrix of Decision Tree Classifier"""

cm = metrics.confusion_matrix(y_test, y_pred_dtree, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

"""#### AUC of DecisionTree Model"""

y_pred_proba = dTree.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

dt_auc = auc
dt_auc

"""### Under Sample Decision Tree"""

dtree_grid_search(X_train_down, y_train_down)

dTree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 6)
dTree.fit(X_train_down, y_train_down)

print(dTree.score(X_train_down,y_train_down))
print(dTree.score(X_test,y_test))

y_pred_dtree = dTree.predict(X_test)

print(metrics.classification_report(y_test, y_pred_dtree))

cm = metrics.confusion_matrix(y_test, y_pred_dtree, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

y_pred_proba = dTree.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

dt_tacc_down = dTree.score(X_test,y_test)
dt_train_acc_down = dTree.score(X_train_down, y_train_down)
dt_auc_down = auc
dt_auc_down

"""## Ensemble learning - AdaBoosting"""

def ada_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = {'n_estimators':[10, 50, 100, 500], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # AdaBoost model
    ada = AdaBoostClassifier()
    
    # Use gridsearch to test all values
    ada_gscv = GridSearchCV(ada, param_grid, n_jobs=-1, cv=cv, scoring='accuracy')
    #fit model to data
    grid_result = ada_gscv.fit(X, y)
    
    return ada_gscv.best_params_

ada_grid_search(X_train, y_train)

"""### Over Sample AdaBoost"""

abcl = AdaBoostClassifier(n_estimators=500, learning_rate = 0.1)
abcl = abcl.fit(X_train, y_train)

y_pred_abcl = abcl.predict(X_test)

print(abcl.score(X_train, y_train))
print(abcl.score(X_test,y_test))

print(metrics.classification_report(y_test, y_pred_abcl))

ada_train_acc = abcl.score(X_train, y_train)
ada_tacc = abcl.score(X_test,y_test)

"""#### Confusion Matrix AdaBoosting model"""

cm = metrics.confusion_matrix(y_test, y_pred_abcl, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

"""#### AUC of Adaboosting model"""

y_pred_proba = abcl.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

ada_auc = auc
ada_auc

"""### Under Sample AdaBoost"""

ada_grid_search(X_train_down, y_train_down)

abcl = AdaBoostClassifier(n_estimators=100, learning_rate = 0.1)
abcl = abcl.fit(X_train_down, y_train_down)

y_pred_abcl = abcl.predict(X_test)

print(abcl.score(X_train_down, y_train_down))
print(abcl.score(X_test,y_test))

print(metrics.classification_report(y_test, y_pred_abcl))

cm = metrics.confusion_matrix(y_test, y_pred_abcl, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

y_pred_proba = abcl.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

ada_train_acc_down = abcl.score(X_train_down, y_train_down)
ada_tacc_down = abcl.score(X_test,y_test)
ada_auc_down = auc
ada_auc_down

"""## Random forest classifier"""

def rf_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = { 
    'n_estimators': [5,10,20,40,50,60,70,80,100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
    }
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    # Random Forest model
    rf = RandomForestClassifier()
    
    #use gridsearch to test all values
    rf_gscv = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    #fit model to data
    rf_gscv.fit(X, y)
    
    return rf_gscv.best_params_

rf_grid_search(X_train, y_train)

"""### Over Sample Random Forest"""

rfcl = RandomForestClassifier(n_estimators=70, max_features='sqrt', max_depth=7, criterion='entropy')
rfcl = rfcl.fit(X_train, y_train)

y_pred_rf = rfcl.predict(X_test)

print(rfcl.score(X_train,y_train))
print(rfcl.score(X_test,y_test))

print(metrics.classification_report(y_test, y_pred_rf))

rf_tacc = rfcl.score(X_test,y_test)
rf_train_acc = rfcl.score(X_train, y_train)

"""#### Confusion matrix of Random Forest Classifier Model"""

cm = metrics.confusion_matrix(y_test, y_pred_rf, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

"""#### AUC of Random Forest Classifier Model"""

y_pred_proba = rfcl.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

rf_auc = auc
rf_auc

"""### Under Sample Random Forest"""

rf_grid_search(X_train_down, y_train_down)

rfcl = RandomForestClassifier(n_estimators=80, max_features='log2', max_depth=7, criterion='entropy')
rfcl = rfcl.fit(X_train_down, y_train_down)

y_pred_rf = rfcl.predict(X_test)

print(rfcl.score(X_train_down,y_train_down))
print(rfcl.score(X_test,y_test))

print(metrics.classification_report(y_test, y_pred_rf))

cm = metrics.confusion_matrix(y_test, y_pred_rf, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

y_pred_proba = rfcl.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

rf_tacc_down = rfcl.score(X_test,y_test)
rf_train_acc_down = rfcl.score(X_train_down, y_train_down)
rf_auc_down = auc
rf_auc_down

"""## kNN"""

def knn_grid_search(X, y):
    #create a dictionary of all values we want to test
    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    knn = KNeighborsClassifier()
    
    #use gridsearch to test all values
    knn_gscv = GridSearchCV(knn, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    #fit model to data
    knn_gscv.fit(X, y)
    
    return knn_gscv.best_params_

knn_grid_search(X_train, y_train)

"""### Over Sample kNN"""

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

print(knn.score(X_train, y_train))
print(knn.score(X_test, y_test))

print(metrics.classification_report(y_test, y_pred_knn))

knn_tacc = knn.score(X_test, y_test)
knn_train_acc = knn.score(X_train, y_train)

"""#### Confusion Matrix of kNN"""

cm = metrics.confusion_matrix(y_test, y_pred_knn, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

"""#### AUC of kNN"""

y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

knn_auc= auc
knn_auc

"""### Under Sample kNN"""

knn_grid_search(X_train_down, y_train_down)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_down, y_train_down)

y_pred_knn = knn.predict(X_test)

print(knn.score(X_train_down, y_train_down))
print(knn.score(X_test, y_test))

print(metrics.classification_report(y_test, y_pred_knn))

cm = metrics.confusion_matrix(y_test, y_pred_knn, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

y_pred_proba = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

knn_tacc_down = knn.score(X_test, y_test)
knn_train_acc_down = knn.score(X_train_down, y_train_down)
knn_auc_down = auc
knn_auc_down

"""## SVM"""

def svm_grid_search(X, y):
    #create a dictionary of all values we want to test
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001, 0.4, 0.2, 0.8],'kernel': ['rbf', 'poly', 'sigmoid']}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    svm = SVC()
    
    #use gridsearch to test all values
    svm_gscv = RandomizedSearchCV(estimator = svm,
                           param_distributions = param_grid,
                           scoring = 'accuracy',
                           cv = cv,
                           n_jobs = -1)
    #fit model to data
    svm_gscv.fit(X, y)
    
    return svm_gscv.best_params_

svm_grid_search(X_train, y_train)

"""### Over Sample SVM"""

from sklearn import svm
svm = SVC(gamma=0.8, C=10, kernel='rbf', probability=True)

svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)

print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))

print(metrics.classification_report(y_test, y_pred_svm))

svm_tacc = svm.score(X_test, y_test)
svm_train_acc = svm.score(X_train, y_train)

"""#### Confusion Matrix of SVM"""

cm = metrics.confusion_matrix(y_test, y_pred_svm, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

"""#### AUC of SVM"""

y_pred_proba = svm.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

svm_auc = auc
svm_auc

"""### Under Sample SVM"""

svm_grid_search(X_train_down, y_train_down)

from sklearn import svm
svm = SVC(gamma=0.4, C=1, kernel='rbf', probability=True)

svm.fit(X_train_down, y_train_down)

y_pred_svm = svm.predict(X_test)

print(svm.score(X_train_down, y_train_down))
print(svm.score(X_test, y_test))

print(metrics.classification_report(y_test, y_pred_svm))

cm = metrics.confusion_matrix(y_test, y_pred_svm, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

y_pred_proba = svm.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

svm_tacc_down = svm.score(X_test, y_test)
svm_train_acc_down = svm.score(X_train_down, y_train_down)
svm_auc_down = auc
svm_auc_down

"""## XGBoost Model"""

def xgb_grid_search(X, y):
    # Create a dictionary of all values we want to test
    param_grid = {
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    xgb = XGBClassifier()
    
    #use gridsearch to test all values
    xgb_gscv =  RandomizedSearchCV(estimator = xgb,
                           param_distributions = param_grid,
                           scoring = 'accuracy',
                           cv = cv,
                           n_jobs = -1)
    #fit model to data
    xgb_gscv.fit(X, y)
    
    return xgb_gscv.best_params_

xgb_grid_search(X_train, y_train)

"""### Over Sample XGBoost"""

xgb = XGBClassifier(min_child_weight=1, max_depth=10, learning_rate=0.25, gamma=0.4, colsample_bytree=0.3)
xgb.fit(X_train,y_train)

y_pred_xgb = xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_xgb))
print("Precision:",metrics.precision_score(y_test, y_pred_xgb))
print("Recall:",metrics.recall_score(y_test, y_pred_xgb))

print(xgb.score(X_train,y_train))
print(xgb.score(X_test,y_test))

xgb_tacc = xgb.score(X_test,y_test)
xgb_train_acc = xgb.score(X_train, y_train)

"""#### Confusion Matrix of XGBoost"""

cm = metrics.confusion_matrix(y_test, y_pred_xgb, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

"""#### AUC of XGBoost Model"""

y_pred_proba = xgb.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

xgb_auc = auc

"""### Under Sample XGBoost"""

xgb_grid_search(X_train_down, y_train_down)

xgb = XGBClassifier(min_child_weight=1, max_depth=8, learning_rate=0.15, gamma=0.1, colsample_bytree=0.7)
xgb.fit(X_train_down,y_train_down)

y_pred_xgb = xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_xgb))
print("Precision:",metrics.precision_score(y_test, y_pred_xgb))
print("Recall:",metrics.recall_score(y_test, y_pred_xgb))

print(xgb.score(X_train_down,y_train_down))
print(xgb.score(X_test,y_test))

cm = metrics.confusion_matrix(y_test, y_pred_xgb, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],
                         columns = [i for i in ["Predict 1", "Predict 0"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')

y_pred_proba = xgb.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
fpr 
tpr

auc = metrics.roc_auc_score(y_test, y_pred_proba)
auc

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

xgb_tacc_down = xgb.score(X_test,y_test)
xgb_train_acc_down = xgb.score(X_train_down, y_train_down)
xgb_auc_down = auc

"""# Comparision of all Models

## Over Sample Models
"""

def comp_model(model_list, model_train_acc_list, model_test_acc_list, model_auc_list):
    data = {"Model Name": model_list, "Train Accuracy(%)": [i*100 for i in model_train_acc_list], "Test Accuracy(%)": [i*100 for i in model_test_acc_list], "AUC Score": model_auc_list}
    Comparision = pd.DataFrame(data)
    return Comparision

model_list = ["Logistic Regression", "Decision Tree Classifier", "AdaBoost", "Random Forest Classifier", "kNN", "SVM", "XGBoost"]
model_train_acc_list = [lr_train_acc, dt_train_acc, ada_train_acc, rf_train_acc, knn_train_acc, svm_train_acc, xgb_train_acc]
model_test_acc_list = [lr_tacc, dt_tacc, ada_tacc, rf_tacc, knn_tacc, svm_tacc, xgb_tacc]
model_auc_list = [lr_auc, dt_auc, ada_auc, rf_auc, knn_auc, svm_auc, xgb_auc]
comp_model(model_list, model_train_acc_list, model_test_acc_list, model_auc_list)

"""- The higher the AUC, the better the performance of the model at distinguishing between the positive and negative classes
- We can say that Random Forest Classifier Model and AdaBoost Model are good for our over sampled dataset as it is giving highest AUC score as well as highest accuracy.
- Lets do Cross Validation and find out which model is Best

## Under Sample Models
"""

model_list = ["Logistic Regression", "Decision Tree Classifier", "AdaBoost", "Random Forest Classifier", "kNN", "SVM", "XGBoost"]
model_train_acc_list = [lr_train_acc_down, dt_train_acc_down, ada_train_acc_down, rf_train_acc_down, knn_train_acc_down, svm_train_acc_down, xgb_train_acc_down]
model_test_acc_list = [lr_tacc_down, dt_tacc_down, ada_tacc_down, rf_tacc_down, knn_tacc_down, svm_tacc_down, xgb_tacc_down]
model_auc_list = [lr_auc_down, dt_auc_down, ada_auc_down, rf_auc_down, knn_auc_down, svm_auc_down, xgb_auc_down]
comp_model(model_list, model_train_acc_list, model_test_acc_list, model_auc_list)



### Random Forest Classfier


skfold = StratifiedKFold(n_splits=5)
model = RandomForestClassifier(n_estimators=70, max_features='sqrt', max_depth=7, criterion='entropy')
scores = cross_val_score(model, features, y, cv=skfold)

print(scores)
print(np.mean(scores))

"""### AdaBoost Classifier"""

skfold = StratifiedKFold(n_splits=5)
model = AdaBoostClassifier(n_estimators=500, learning_rate = 0.1)
scores = cross_val_score(model, features, y, cv=skfold)

print(scores)
print(np.mean(scores))

"""### XGBoost Classifier"""

skfold = StratifiedKFold(n_splits=5)
model = XGBClassifier(min_child_weight=1, max_depth=10, learning_rate=0.25, gamma=0.4, colsample_bytree=0.3)
scores = cross_val_score(model, features, y, cv=skfold)

print(scores)
print(np.mean(scores))

"""## Over Sample CV"""

len(y[y==0]), len(y[y==1])

os =  RandomOverSampler(sampling_strategy=1)

X_train, y_train = os.fit_resample(features, y)

print(len(y_train[y_train==0]), len(y_train[y_train==1]))
print(len(X_train))

"""### Random Forest Classifier"""

skfold = StratifiedKFold(n_splits=5)
model = RandomForestClassifier(n_estimators=70, max_features='sqrt', max_depth=7, criterion='entropy')
scores = cross_val_score(model, X_train, y_train, cv=skfold)

print(scores)
print(np.mean(scores))

"""### AdaBoost Model"""

skfold = StratifiedKFold(n_splits=5)
model = AdaBoostClassifier(n_estimators=500, learning_rate = 0.1)
scores = cross_val_score(model, X_train, y_train, cv=skfold)

print(scores)
print(np.mean(scores))

"""### XGBoost Model"""

skfold = StratifiedKFold(n_splits=5)
model = XGBClassifier(min_child_weight=1, max_depth=10, learning_rate=0.25, gamma=0.4, colsample_bytree=0.3)
scores = cross_val_score(model, X_train, y_train, cv=skfold)

print(scores)
print(np.mean(scores))

"""## Under Sample CV"""

from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()

X_train_down,y_train_down = rus.fit_resample(features, y)

print(len(y_train_down[y_train_down==0]), len(y_train_down[y_train_down==1]))
print(len(X_train_down))

"""### XGBoost Classifier"""

skfold = StratifiedKFold(n_splits=5)
model = XGBClassifier(min_child_weight=1, max_depth=8, learning_rate=0.15, gamma=0.1, colsample_bytree=0.7)
scores = cross_val_score(model, X_train_down,y_train_down, cv=skfold)

print(scores)
print(np.mean(scores))

"""### Decision Tree Classifier"""

skfold = StratifiedKFold(n_splits=5)
model = DecisionTreeClassifier(criterion = 'entropy', max_depth = 6)
scores = cross_val_score(model, X_train_down,y_train_down, cv=skfold)

print(scores)
print(np.mean(scores))

"""### Random Forest Classifier"""

skfold = StratifiedKFold(n_splits=5)
model = RandomForestClassifier(n_estimators=80, max_features='log2', max_depth=7, criterion='entropy')
scores = cross_val_score(model, X_train_down, y_train_down, cv=skfold)

print(scores)
print(np.mean(scores))

"""**Observations:**
- From this Cross Validation, we can conclude that **Random Forest Classifier model** is best for our project. Also model showing hgiher accuracy and auc score in **over sampling**.

# Building the Prediction System - Random Forest Classifier
"""

#input data and transform into numpy array
in_data= np.asarray(tuple(map(float,input().rstrip().split(','))))

#reshape and scale the input array
in_data_re = in_data.reshape(1,-1)
in_data_sca = scaler.transform(in_data_re)

#print the predicted output for input array
print("Chronic Kidney Disease Detected" if rfcl.predict(in_data_sca) else "Chronic Kidney Disease Not Detected")

"""***Extra data on which you can try our both Prediction System***

Chronic Disease Positive:
- [48, 80, 1, 0, 1, 1, 0, 0, 121, 36, 1.2, 111, 15.4, 7800, 5.2, 1, 1, 0, 1, 0, 0]
- [7, 50, 4, 0, 1, 1, 0, 0, 121, 18, 0.8, 111, 11.3, 6000, 5.2, 0, 0, 0, 1, 0, 0]

Chronic Disease Negative:
- [40, 80, 0, 0, 1, 1, 0, 0, 140, 10, 1.2, 135, 15, 10400, 4.5, 0, 0, 0, 1, 0, 0]
- [23, 80, 0, 0, 1, 1, 0, 0, 70, 36, 1, 150, 17, 9800, 5, 0, 0, 0, 1, 0, 0]

# Multilayer Neural Network
"""

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from keras.layers import Dense, Activation, LeakyReLU, Dropout
from keras.activations import relu, sigmoid

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_curve

"""## Hyperparameter Tuning

**Hyperparameters**
- How many number of hidden layers we should have?
- How many number of neurons we should have in hidden layers?
- Learning Rate
"""

X_train.shape

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units_inp', min_value=32, max_value=512, step=32),
                               activation=hp.Choice( 'activation', ['tanh', 'relu', 'LeakyReLU', 'elu']),
                               input_dim = 21
                          )
              )
    
    for i in range(hp.Int('num_layers', 2, 20)):                 # Number of layers
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,       # Number of neuron, here it is 32-512
                                            step=32),
                               activation=hp.Choice( 'activation', ['tanh', 'relu', 'LeakyReLU', 'elu'])))
                               
        if hp.Boolean("dropout"):
            model.add(layers.Dropout(rate=0.25))
        
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy'])
    
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='MultiNN',
    project_name='Kidney Disease Detection')

tuner.search_space_summary()

tuner.search(X_train, y_train, epochs=5, validation_split=0.2)

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]

best_model.summary()

best_model.fit(X_train,y_train,epochs=50)

# Train and Test accuracy
scores = best_model.evaluate(X_train,y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))
scores = best_model.evaluate(X_test,y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

y_test_pred_probs = best_model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test,y_test_pred_probs)

auc = metrics.roc_auc_score(y_test, y_test_pred_probs)
auc

plt.plot(FPR,TPR,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.plot([0,1],[0,1],'--',color='black')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

"""**Conclusion:**
- Using Dense Neural Network, we are getting almost 100% training accuracy and 100% test accuracy which seems to be very good.
- Hence both Dense Neural Network and Random Forest Model is best for this project to predict whether a patient has CKD or not
"""