# Income prediction model
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

data=pd.read_excel("HR.xlsx")

#replace the object items in coloumn with "NaN"
data['Total_Sales'] = data['Total_Sales'].replace(' ', pd.NA)
#change the dtype of "Total_Sales" as float
data['Total_Sales'] = data['Total_Sales'].astype('Float64')
data['Total_Sales'] = data['Total_Sales'].astype('float64')

#missing value handling
# Forward-fill missing values from index 124 to 135
data['Base_pay'] = data['Base_pay'].fillna(method='ffill', limit=12)
# Backward-fill missing values from index 136 to 146
data['Base_pay'] = data['Base_pay'].fillna(method='bfill', limit=11)
for feature  in ['openingbalance','Total_Sales']:
    data[feature ]=data[feature ].fillna(data[feature ].median())

#encoding
#['Dependancies','Calls','Billing','Rating']
df = data[['Education','Gender','Type']]
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first')
one_hot_encoded = encoder.fit_transform(df)

# Saving model to disk
pickle.dump(one_hot_encoded, open('encoder.pkl','wb'))


one_hot_encoded_array = one_hot_encoded.toarray()
columns = encoder.get_feature_names_out(input_features= ['Education', 'Gender', 'Type'])
df_encoded = pd.DataFrame(one_hot_encoded_array, columns=columns)

#scaling
x = data.drop(['Gender','Dependancies','Calls','Type','Billing','Rating','Education','Salary','Business'], axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled = scaler.fit_transform(x)


scaled =pd.DataFrame(scaled,columns=x.columns)

#feature reduction
from sklearn.decomposition import PCA
pca = PCA(0.99)
s_pca = pca.fit_transform(scaled)
selected_columns = pca.components_
selected_columns_names = scaled.columns[np.argmax(np.abs(selected_columns), axis=1)]
selected_columns_names =np.unique(selected_columns_names)
data1  = pd.concat([scaled[selected_columns_names],df_encoded,data['Salary']], axis=1)

# Saving model to disk
scaler = StandardScaler()
scaled = scaler.fit_transform(x[selected_columns_names])
pickle.dump(scaler, open('scaler.pkl','wb'))



#data splitting
y = data1['Salary']
X = data1.drop(['Salary'],axis=1)
#split the data into training and testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state = 42)
#from sklearn.linear_model import LinearRegression
#lr = LinearRegression()
#Linear_Model = lr.fit(X_train,y_train)
#build a model
from sklearn.ensemble import GradientBoostingRegressor
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_regressor.fit(X_train, y_train)


# Saving model to disk
pickle.dump(gb_regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
