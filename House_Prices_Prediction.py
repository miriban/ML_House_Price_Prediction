"""
    Houses Prices Prediction using GraphLab Create
    Created By: Mohammed J. AbuIriban
"""
import graphlab

# load data ( avaliable on https://d396qusza40orc.cloudfront.net/phoenixassets/course1-for-students/home_data.gl.zip)
sales = graphlab.SFrame("home_data.gl/")

# split our data into train data and test data
train_data, test_data = sales.random_split(0.8,seed=0) # 80% is train data and 20% is test data

# determine which features we are going to use for our ML Model
ml_features =\
[
'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode',
'condition', # condition of house
'grade', # measure of quality of construction
'waterfront', # waterfront property
'view', # type of view
'sqft_above', # square feet above ground
'sqft_basement', # square feet in basement
'yr_built', # the year built
'yr_renovated', # the year renovated
'lat', 'long', # the lat-long of the parcel
'sqft_living15', # average sq.ft. of 15 nearest neighbors
'sqft_lot15', # average lot size of 15 nearest neighbors
]

# Mahcine Learning model
ml_model = graphlab.linear_regression.create(train_data,target="price",features=ml_features)

# Testing the model
print "The Real Average Prices is => ",test_data['price'].mean()
print "The ML Model Average Prices is => ",ml_model.evaluate(test_data)

# Predicting Houses using the model
house = sales[sales['id']=='3530450100']
print "Real Price of Target House is => ",house['price']
print "Predicted Price Of Target House => ",ml_model.predict(house)
