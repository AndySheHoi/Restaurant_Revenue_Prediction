import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.isnull().sum()
test.isnull().sum()

train.duplicated().sum()
test.duplicated().sum()

cat_feature = [col for col in train.columns if train[col].dtypes == "O"]
num_feature = [col for col in train.columns if train[col].dtypes != "O"]

plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.boxplot(train.revenue,color='orange')
plt.subplot(1,2,2)
sns.distplot(train.revenue, bins=20, kde=False,color='purple')
plt.show()

rev_stat=train.revenue.describe()

# calculating interquartile range
iqr=rev_stat['75%']-rev_stat['25%']
upper=rev_stat['75%']+1.5*iqr
lower=rev_stat['25%']-1.5*iqr

print('\nLower bound: {} \nUpper bound: {}'.format(lower,upper))

# remove outliers
train[train.revenue>upper]

# split datetime
train_date=train['Open Date'].str.split('/', n = 2, expand = True)
train['month']=train_date[0]
train['days']=train_date[1]
train['year']=train_date[2]

test_date=test['Open Date'].str.split('/', n = 2, expand = True)
test['month']=test_date[0]
test['days']=test_date[1]
test['year']=test_date[2]

def featureCatPlot(col):
    plt.figure(figsize=(20,5))
    i=1
    if not train[col].dtype.name=='int64' and not train[col].dtype.name=='float64':
        plt.subplot(1,2,i)
        sns.boxplot(x=col,y='revenue',data=train)
        plt.xticks(rotation=60)
        plt.ylabel('Revenue')
        i+=1 
        plt.subplot(1,2,i)
        mean=train.groupby(col)['revenue'].mean()
        level=mean.sort_values().index.tolist()
        train[col]=train[col].astype('category')
        train[col].cat.reorder_categories(level,inplace=True)
        train[col].value_counts().plot()
        plt.xticks(rotation=60)
        plt.xlabel(col)
        plt.ylabel('Counts')       
        plt.show()

#  plot correlation between revenue and continuous feature
def numFeaturePlot():
    features=(train.loc[:,'P1':'P37']).columns.tolist()
    plt.figure(figsize=(35,38))
    j=1
    while j<len(features):
        col=features[j-1]
        plt.subplot(9,4,j)
        sorted_grp = train.groupby(col)["revenue"].sum().sort_values(ascending=False).reset_index()
        x_val = sorted_grp.index
        y_val = sorted_grp['revenue'].values
        plt.scatter(x_val, y_val)
        plt.xticks(rotation=60)
        plt.xlabel(col, fontsize=20)
        plt.ylabel('Revenue', fontsize=20)
        j+=1    
    plt.tight_layout()
    plt.show()

numFeaturePlot()

for feat in cat_feature:
    if feat != 'Open Date':
        featureCatPlot(feat)

featureCatPlot('month')

# Plotting heatmap between revenue and P variables
fig=plt.figure(figsize=(45,32))
features=(train.loc[:,'P1':'P37']).columns.tolist()
sns.heatmap(train[features+['revenue']].corr(),cmap='Greens',annot=True)
plt.xticks(rotation=45)
plt.show()

features=(train.loc[:,'P1':'P37']).columns.tolist()
train[features].hist(figsize=(20,15))
plt.show()


# =============================================================================
# Feature Engineering
# =============================================================================

# Square root of the P variables
ntrain= np.sqrt(train.loc[:,'P1':'P37'])
ntest= np.sqrt(test.loc[:,'P1':'P37'])

col_train = ntrain.columns
col_test = ntest.columns

SS =StandardScaler()

ntrain = SS.fit_transform(ntrain)
ntest = SS.fit_transform(ntest)

ntrain= pd.DataFrame(ntrain,columns=col_train)
ntest= pd.DataFrame(ntest,columns=col_test)

train = train[train.columns[train.columns.isin(['Id','City','City Group','Type','month','days','year','revenue'])]]
train = pd.concat([train,ntrain],axis=1)
test = test[test.columns[test.columns.isin(['Id','City','City Group','Type','month','days','year'])]]
test = pd.concat([test,ntest],axis=1)

city = train.groupby('City')['revenue'].agg(['size','count','min','max','mean']).sort_values(by='mean',ascending = False)
city.columns = ['no. of rows','rows with revenue','min','max','average revenue']

x = city.index[:10]
city_x = [x * 1.0 for x, _ in enumerate(x)]
y_tr = city['no. of rows'][:10]
y_rr = city['rows with revenue'][:10]
y_min = city['min'][:10]
y_max = city['max'][:10]
y_avgr = city['average revenue'][:10]


# =============================================================================
# Model Training and Performance Evaluation
# =============================================================================

y_train = train['revenue']
x_train = train.drop(columns=['revenue'],axis=1)
x_test = test

le =LabelEncoder()
for col in x_train.columns:
    if ((x_train[col].dtype.name == 'object') or (x_train[col].dtype.name == 'category')):
        x_train[col] = le.fit_transform(x_train[col])
        x_test[col] = le.fit_transform(x_test[col])
        
test_label=pd.read_csv('sampleSubmission.csv') 


# Functionalize model fittting
from math import sqrt
mse_list=dict()
gpred=[]

def FitModel(X,Y,algo_name,algorithm,gridSearchParams,cv):
    global gpred
    np.random.seed(10)
    x_train_L,x_test_L,y_train_L,y_test_L = train_test_split(X,Y, test_size = 0.05)
    
    grid = GridSearchCV(
        estimator=algorithm,
        param_grid=gridSearchParams,
        cv=cv,  verbose=1, n_jobs=-1)
    
    grid_result = grid.fit(x_train_L, y_train_L)
    best_params = grid_result.best_params_
    pred = grid_result.predict(x_test_L)
    
    label_list=test_label['Prediction'].tolist()
    
    print('Best Params :',best_params)
    print('Root Mean squared error {}'.format(sqrt(mean_squared_error(y_test_L, pred))))
    
    pred_test = grid_result.predict(x_test)
    gpred= pred_test
    diff = label_list - pred_test
    res_df = pd.concat([pd.Series(pred_test),pd.Series(label_list),pd.Series(diff)],axis=1)
    res_df.columns = ['Prediction','Original Data','Diff']
    print('******************** MSE BASED on ORIGINAL TEST DATA **************************')
    print('Root Mean squared error {}'.format(sqrt(mean_squared_error(label_list, pred_test))))
    mse_list[algo_name]=sqrt(mean_squared_error(label_list, pred_test))
    print('******************** Prediction vs ORIGINAL TEST DATA **************************')
    print(res_df.head())
    
    return gpred, label_list
    
    
pd.options.display.float_format = '{:.2f}'.format

# Random Forest
param_rf ={
            'n_estimators': [50,100,150, 300,500, 700,1000, 2000],
           
        }
gpred_rf, label_list_rf = FitModel(x_train,y_train,'Random Forest',RandomForestRegressor(),param_rf,cv=5)

# SVR
param_svr ={
            'C': [0.1, 1, 100, 1000],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        }
gpred_svr, label_list_svr = FitModel(x_train,y_train,'SVR',SVR(),param_svr,cv=5)

# Gradient Boosting

param_gb ={
            'max_depth':[2,3,4,5]
        }
gpred_gb, label_list_gb = FitModel(x_train,y_train,'GradientBoostingRegressor',GradientBoostingRegressor(),param_gb,cv=5)


param_xgb ={
            'max_depth':[2,3,4,5]
        }
gpred_xgb, label_list_xgb = FitModel(x_train,y_train,'XGBRegressor',XGBRegressor(),param_gb,cv=5)


gpred = gpred_svr
label_list = label_list_svr


# =============================================================================
# Semi-Supervised Learning
# =============================================================================

X = pd.concat([x_train,x_test],axis=0)
Y = pd.concat([pd.Series(y_train),pd.Series(gpred)],axis=0)
Y_label =  pd.concat([pd.Series(y_train),pd.Series(label_list)],axis=0)

mse_list_ss=dict()


def FitModelSS(X,Y,algo_name,algorithm,gridSearchParams,cv):
    np.random.seed(10)
    # 100137 * 0.8 = 80110
    x_train = X[0:80110]
    x_test = X[80110:]
    y_train = Y[0:80110]
    y_test= Y_label[80110:]
    
    grid = GridSearchCV(
        estimator=algorithm,
        param_grid=gridSearchParams,
        cv=cv,  verbose=1, n_jobs=-1)
    
    
    grid_result = grid.fit(x_train, y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict(x_test)

    print('Best Params :',best_params)
    print('Root Mean squared error {}'.format(sqrt(mean_squared_error(y_test, pred))))
    
    diff = y_test - pred
    res_df = pd.concat([pd.Series(pred),pd.Series(y_test),pd.Series(diff)],axis=1)
    res_df.columns = ['Prediction','Original Data','Diff']
    
    mse_list_ss[algo_name]=sqrt(mean_squared_error(y_test, pred))
    print('******************** Prediction vs ORIGINAL TEST DATA **************************')
    print(res_df.head())


# Random Forest
FitModelSS(X,Y,'Random Forest',RandomForestRegressor(),param_rf,cv=5)

# SVR
FitModelSS(X,Y,'SVR',SVR(),param_svr,cv=5)

# Gradient Boosting
FitModelSS(X,Y,'GradientBoostingRegressor',GradientBoostingRegressor(),param_gb,cv=5)

# XGBoost
FitModelSS(X,Y,'XGBoostRegressor',XGBRegressor(),param_xgb,cv=5)


print('mse_list:\n', mse_list)
print('mse_list_ss:\n',mse_list_ss)