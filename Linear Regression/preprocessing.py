# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 13:35:01 2025

@author: advit
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler

data=pd.read_csv("housing.csv")
data["bedrooms_per_room"]=data["total_bedrooms"]/data["total_rooms"]
encoder=OneHotEncoder(sparse_output=False)
selected_features=data[["median_income","bedrooms_per_room","ocean_proximity"]]
ocean_prox_encoded=encoder.fit_transform(selected_features[["ocean_proximity"]])

encoded_df=pd.DataFrame(ocean_prox_encoded,columns=encoder.get_feature_names_out(["ocean_proximity"]))

final_df=pd.concat([selected_features[["median_income","bedrooms_per_room"]],encoded_df],axis=1)

final_df['target']=data['median_house_value']

train_set,test_set=train_test_split(final_df, test_size=0.2,random_state=42)

num_features=["median_income","bedrooms_per_room","target"]
scaler=StandardScaler()

scaled_train_num=scaler.fit_transform(train_set[num_features])
scaled_test_num=scaler.transform(test_set[num_features])

print("Means for columns:", num_features)
print(scaler.mean_)

print("Stds for columns:", num_features)
print(scaler.scale_)

trained_scaled_df=train_set.copy()
test_scaled_df=test_set.copy()

trained_scaled_df[num_features]=scaled_train_num
test_scaled_df[num_features]=scaled_test_num



train_set.to_csv("train_data.csv",index=False)
test_set.to_csv("test_data.csv",index=False)
trained_scaled_df.to_csv('train_normalised2.csv',index=False)
test_scaled_df.to_csv('test_normlaised2.csv',index=False)