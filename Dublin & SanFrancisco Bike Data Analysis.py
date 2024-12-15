#!/usr/bin/env python
# coding: utf-8

# 
# # <span style="color: blue;">Dublin & SanFrancisco Bike Data Analysis</span>  

# ## LIBRARIES

# In[10]:


import pandas as pd #Pandas is a powerful library for data manipulation and analysis, especially for structured data in tabular form.
import numpy as np #Numpy is widely used for numerical operations and handling large multidimensional arrays and matrices.

import matplotlib.pyplot as plt #Matplotlib is a core library for creating static, interactive, and animated visualizations in Python.
import seaborn as sns #Seaborn builds on matplotlib and provides a high-level interface for drawing attractive statistical graphics.

import scipy.stats as stats #SciPy's stats module provides a wide variety of statistical functions for analysis, hypothesis testing, and probability distributions.

from sklearn.preprocessing import LabelEncoder #LabelEncoder is used for converting categorical variables into numeric form by encoding each unique category as a different integer.
from sklearn.preprocessing import scale #Scale standardizes the features by adjusting their values to ensure equal importance when used in machine learning models.
from sklearn.preprocessing import StandardScaler #StandardScaler rescales features to have zero mean and unit variance, ensuring that all features contribute equally to model performance.
from sklearn import metrics #Metrics provides various functions for evaluating the performance of machine learning models, such as accuracy and F1 score.
from sklearn.metrics import confusion_matrix #Confusion matrix helps to assess the performance of classification algorithms by showing true vs. predicted class counts.
from sklearn.metrics import precision_score #Precision score measures the accuracy of positive predictions, reflecting the proportion of true positives among all positive predictions.
from sklearn.metrics import recall_score #Recall score shows the model’s ability to correctly identify all relevant instances, focusing on positive class predictions.
from sklearn.model_selection import train_test_split #train_test_split is a method for splitting datasets into training and testing subsets for model evaluation.
from sklearn.model_selection import KFold #KFold is a cross-validation technique that splits data into K subsets to validate the model on different splits to avoid overfitting.
from sklearn.model_selection import cross_val_score #cross_val_score performs cross-validation to assess the model’s performance by training it on different data splits.
from sklearn.model_selection import GridSearchCV #GridSearchCV automates hyperparameter tuning by exhaustively searching for the best combination of parameters.
from sklearn.decomposition import PCA #PCA (Principal Component Analysis) reduces the dimensionality of the dataset, identifying the most important features.

#Machine learning algorithms libraries
from sklearn.svm import SVC #SVC (Support Vector Classifier) is a supervised machine learning algorithm used for classification tasks.
from sklearn.cluster import KMeans #KMeans is an unsupervised clustering algorithm used for grouping data into clusters based on similarity.
from sklearn.linear_model import LogisticRegression #Logistic Regression is a linear model for binary or multi-class classification tasks.
from sklearn.neighbors import KNeighborsClassifier #K-Nearest Neighbors (KNN) is a simple and intuitive classification algorithm based on feature proximity.
from sklearn.naive_bayes import GaussianNB #Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem with the assumption that features are normally distributed.
from sklearn.tree import DecisionTreeClassifier #Decision Trees create a flowchart-like structure to make decisions based on feature values for classification tasks.
from sklearn.ensemble import RandomForestClassifier #Random Forest is an ensemble method that creates a collection of decision trees and combines their predictions to improve accuracy.
import json #The json library is used for working with data in JSON format, such as parsing or generating JSON.
import requests #The requests library allows for making HTTP requests, enabling communication with APIs and web resources.

import pprint #PrettyPrint makes the output more readable and structured, especially useful for printing complex data structures.
 
import warnings #Warnings is used to control the display of warnings that may arise during code execution.
warnings.filterwarnings("ignore") #This command suppresses warning messages to keep the output clean.


# --------------------------------------------------------------------------------------------------------------------------------

# ## FUNCTION

# In[11]:


# The following functions are defined to be used later for various tasks

def label_graph(ticksfont, x_label, y_label, title_label, fontsize):
    # This function customizes the graph's labels, ticks, and title for better readability and presentation.
    plt.xticks(fontsize=ticksfont)  # Adjust the font size of x-axis ticks.
    plt.yticks(fontsize=ticksfont)  # Adjust the font size of y-axis ticks.
    plt.xlabel(x_label, fontsize=fontsize)  # Set the label for the x-axis with the specified font size.
    plt.ylabel(y_label, fontsize=fontsize)  # Set the label for the y-axis with the specified font size.
    plt.title(title_label, fontsize=fontsize)  # Set the graph title with the specified font size.

def median(array):
    # This function calculates and prints the median of an input array (middle value of the sorted data).
    median = np.median(array)  # Compute the median using numpy's median function.
    return print("Median:", median)  # Print the computed median value.

def mean(array):
    # This function calculates and prints the mean (average) of an input array.
    mean = np.mean(array)  # Compute the mean using numpy's mean function.
    return print("Mean:", mean)  # Print the computed mean value.

def variance(array):
    # This function calculates and prints the variance (spread of data) of an input array.
    variance = np.var(array)  # Compute the variance using numpy's var function.
    return print("Variance:", variance)  # Print the computed variance value.


# --------------------------------------------------------------------------------------------------------------------------------

# ## DATAFRAME

# ### SAN FRANCISCO BIKES 2020 LAST QUARTER ( OCT-NOV-DEC ) Data from API

# In[12]:


# Sending a GET request to the API URL to retrieve data from the source
sf_BikeApiReq = requests.get("https://data.sfgov.org/resource/7jbp-yzp3.json").text  # Fetch data from the San Francisco bike-sharing API.

# The received text data (JSON format) is parsed into a Python dictionary
sf_BikeJson = json.loads(sf_BikeApiReq)  # Convert the JSON string into a Python dictionary for further manipulation.

# Pretty print is used to enhance the readability of the JSON data, displaying it in a structured format
pprint.pprint(sf_BikeJson)  # Output the parsed JSON data with a cleaner, more readable structure.


# In[13]:


# Converting the JSON data (Python dictionary) into a pandas DataFrame for structured analysis and manipulation
sf_BikeJsonDF = pd.DataFrame.from_dict(sf_BikeJson)  # Convert the JSON data into a DataFrame for easier access and handling of tabular data.

# we can print the DataFrame using pprint to display it in a more readable format (commented out here)
pprint.pprint(sf_BikeJsonDF)  # Uncomment this line to view the DataFrame with improved readability.




# In[14]:


# Display the DataFrame to inspect its contents (this would display the DataFrame in an interactive environment like Jupyter)
sf_BikeJsonDF  # Uncomment this line to display the DataFrame.


# In[15]:


#First 5 (default) lines viewed
sf_BikeJsonDF.head(5)


# In[16]:


#Total number of columns and rows were checked
sf_BikeJsonDF.shape


# In[17]:


#Last 5 (default) lines viewed
sf_BikeJsonDF.tail(5)


# In[18]:


#The names of the columns were learned
sf_BikeJsonDF.columns


# --------------------------------------------------------------------------------------------------------------------------------

# ### SAN FRANCISCO BIKES 2020 LAST QUARTER ( OCT-NOV-DEC ) Data from CSV

# In[19]:


# Loading the CSV data files for October, November, and December 2020 into separate DataFrames
sanbikeoct = pd.read_csv('202010-baywheels-tripdata.csv')  # The trip data for October 2020 is read into the 'sanbikeoct' DataFrame.
sanbikenov = pd.read_csv('202011-baywheels-tripdata.csv')  # The trip data for November 2020 is read into the 'sanbikenov' DataFrame.
sanbikedec = pd.read_csv('202012-baywheels-tripdata.csv')  # The trip data for December 2020 is read into the 'sanbikedec' DataFrame.

# Combining the DataFrames from October, November, and December into a single DataFrame using concatenation
sf_BikeDF = pd.concat([sanbikeoct, sanbikenov, sanbikedec])

# Resetting the index of the combined DataFrame to ensure it starts from 0 and has sequential values
sf_BikeDF.reset_index(drop=True, inplace=True)

# Optionally, pprint can be used to display the DataFrame in a more readable format (commented out here)
# pprint.pprint(sf_BikeDF)  # Uncomment to display the combined DataFrame.

# Display the combined DataFrame with trip data for the three months
sf_BikeDF  # This outputs the concatenated DataFrame.


# In[20]:


# Using the shape function to check the dimensions of the merged DataFrame (number of rows and columns)
sf_BikeDF.shape  # Outputs a tuple representing the number of rows and columns in the combined DataFrame.


# In[21]:


#First 5 (default) lines viewed
sf_BikeDF.head()


# In[22]:


#The names of the columns were learned
sf_BikeDF.columns


# In[23]:


sf_BikeDF.describe() #Dataset is examined


# In[24]:


# To reduce the dataset size, we randomly sample 50,000 rows from the DataFrame. 
# This is done to manage large datasets while maintaining data variety by shuffling the rows.
sf_BikeDF = sf_BikeDF.sample(n=50000)  # Randomly selects 50,000 rows from the DataFrame to work with a manageable subset of the data.


# In[25]:


# Converting the 'started_at' and 'ended_at' columns to datetime format for easier manipulation
sf_BikeDF["started_at"] = pd.to_datetime(sf_BikeDF["started_at"])  # Convert 'started_at' column to datetime type.
sf_BikeDF["started_date"] = sf_BikeDF['started_at'].dt.date  # Extract the date part of 'started_at' and store it in 'started_date'.
sf_BikeDF["started_time"] = sf_BikeDF['started_at'].dt.time  # Extract the time part of 'started_at' and store it in 'started_time'.

sf_BikeDF["ended_at"] = pd.to_datetime(sf_BikeDF["ended_at"])  # Convert 'ended_at' column to datetime type.
sf_BikeDF["ended_date"] = sf_BikeDF['ended_at'].dt.date  # Extract the date part of 'ended_at' and store it in 'ended_date'.
sf_BikeDF["ended_time"] = sf_BikeDF['ended_at'].dt.time  # Extract the time part of 'ended_at' and store it in 'ended_time'.


# In[26]:


# Remove unnecessary columns that will not be used in the analysis
sf_BikeDF = sf_BikeDF.drop(columns=["start_lat", "start_lng", "end_lat", "end_lng", "started_at", "ended_at"])
sf_BikeDF.columns


# In[27]:


# Check the data types of each column in the dataset
sf_BikeDF.dtypes


# In[28]:


# Verify the total number of rows and columns in the dataset
sf_BikeDF.shape


# In[29]:


# Determine the number of non-missing elements in each column
sf_BikeDF.count()


# In[30]:


# Obtain detailed information about the dataset, including column types and non-null counts
sf_BikeDF.info()


# In[31]:


#Last 5 (default) lines viewed
sf_BikeDF.tail()


# In[32]:


# Identify columns with missing values and count the number of null entries in each
sf_BikeDF.isnull().sum()


# In[33]:


# Ignore rows with NaN values in specific columns (optional approach)
# sf_BikeDF[sf_BikeDF[["start_station_name", "start_station_id", "end_station_name", "end_station_id"]].notna()]

# Remove rows with any NaN values from the dataset
sf_BikeDF.dropna(inplace=True)


# In[34]:


# Verify again to ensure there are no remaining NaN values in the dataset
sf_BikeDF.isnull().sum()


# In[35]:


# Generate descriptive statistics for the dataset using the describe function
print(sf_BikeDF['ride_id'].describe())
print("----------")
print(sf_BikeDF['rideable_type'].describe())
print("----------")
print(sf_BikeDF['start_station_name'].describe())
print("----------")
print(sf_BikeDF['start_station_id'].describe())
print("----------")
print(sf_BikeDF['end_station_name'].describe())
print("----------")
print(sf_BikeDF['end_station_id'].describe())
print("----------")
print(sf_BikeDF['member_casual'].describe())
print("----------")
print(sf_BikeDF['started_date'].describe())
print("----------")
print(sf_BikeDF['started_time'].describe())
print("----------")
print(sf_BikeDF['ended_date'].describe())
print("----------")
print(sf_BikeDF['ended_time'].describe())


# In[36]:


# Retrieve unique values from the columns using the unique function
print(sf_BikeDF['ride_id'].unique())
print("----------")
print(sf_BikeDF['rideable_type'].unique())
print("----------")
print(sf_BikeDF['start_station_name'].unique())
print("----------")
print(sf_BikeDF['start_station_id'].unique())
print("----------")
print(sf_BikeDF['end_station_name'].unique())
print("----------")
print(sf_BikeDF['end_station_id'].unique())
print("----------")
print(sf_BikeDF['member_casual'].unique())
print("----------")
print(sf_BikeDF['started_date'].unique())
print("----------")
print(sf_BikeDF['started_time'].unique())
print("----------")
print(sf_BikeDF['ended_date'].unique())
print("----------")
print(sf_BikeDF['ended_time'].unique())


# ### DUBLIN BIKES 2020 LAST QUARTER ( OCT-NOV-DEC ) Data from CSV

# In[37]:


#The file to be processed is assigned to the variable
dublinoct = r"C:\Users\HP\Desktop\LAST BASH"
dublinnov = r"C:\Users\HP\Desktop\LAST BASH"
dublindec = r"C:\Users\HP\Desktop\LAST BASH"
#Csv files read
dublinoct=pd.read_csv('moby-bikes-historical-data-102020.csv') #data is read .
dublinnov=pd.read_csv('moby-bikes-historical-data-112020.csv') #data is read .
dublindec=pd.read_csv('moby-bikes-historical-data-122020.csv') #data is read .
#Dataframes created and merged with concat
dub_BikeDF = pd.concat([dublinoct, dublinnov, dublindec ])
#Index values have been reset
dub_BikeDF.reset_index(drop=True, inplace=True)
#pprint.pprint(dub_BikeDF)
dub_BikeDF


# In[38]:


#First 10 (default) lines viewed
dub_BikeDF.head(10)


# In[39]:


#Total number of columns and rows are showed
dub_BikeDF.shape


# In[40]:


# To reduce the dataset size, we randomly sample 50,000 rows from the DataFrame. 
# This is done to manage large datasets while maintaining data variety by shuffling the rows.
dub_BikeDF = dub_BikeDF.sample(n=50000) # Randomly selects 50,000 rows from the DataFrame to work with a manageable subset of the data.


# In[41]:


#After data regularization ; Total number of columns and rows are showed
dub_BikeDF.shape


# In[42]:


#To provide clearer data, division was made in the dt_start & dt_end columns
dub_BikeDF["dt_start"] = pd.to_datetime(dub_BikeDF["LastRentalStart"])
dub_BikeDF["started_date"] = dub_BikeDF['dt_start'].dt.date
dub_BikeDF["started_time"] = dub_BikeDF['dt_start'].dt.time

dub_BikeDF["dt_end"] = pd.to_datetime(dub_BikeDF["LastGPSTime"])
dub_BikeDF["ended_date"] = dub_BikeDF['dt_end'].dt.date
dub_BikeDF["ended_time"] = dub_BikeDF['dt_end'].dt.time


# In[43]:


dub_BikeDF = dub_BikeDF.drop(['HarvestTime', 'Battery', 'BikeIdentifier',
       'EBikeProfileID', 'EBikeStateID', 'IsMotor', 'IsSmartLock','Latitude', 'Longitude', 'SpikeID','LastGPSTime','LastRentalStart'], axis=1) #Indicated columns are eliminated .


# In[44]:


dub_BikeDF.columns


# In[45]:


#The names of some columns have been changed
newNamesDub =  {"BikeID": "transaction_id", "BikeTypeName": "bike_type"}
dub_BikeDF.rename(columns=newNamesDub, inplace=True)


# In[46]:


unique_bike_types = dub_BikeDF['bike_type'].unique()
print(unique_bike_types)


# In[47]:


#Bike types are named

dub_BikeDF['bike_type'] = dub_BikeDF['bike_type'].replace('DUB-General', 'Classic_Bike')
dub_BikeDF['bike_type'] = dub_BikeDF['bike_type'].replace('Private', 'E_Bike')
dub_BikeDF['bike_type'] = dub_BikeDF['bike_type'].replace('Workshop', 'E_Bike')


# In[48]:


dub_BikeDF.tail(5)


# In[49]:


dub_BikeDF.columns #New coloumns names are viewed


# In[50]:


#Data types of columns were learned
dub_BikeDF.dtypes


# In[51]:


dub_BikeDF.head(10) #Initial 10 lines are showed .


# In[52]:


dub_BikeDF.count() #Data is counted .


# In[53]:


dub_BikeDF.describe() #Data is expressed .


# In[54]:


dub_BikeDF.info 


# In[55]:


#Rows with NaN values were ignored
#dub_BikeDF[dub_BikeDF["station_id"].notna()]
#Rows with #NaN values were deleted
dub_BikeDF.dropna(inplace=True) 


# In[56]:


dub_BikeDF.isnull().sum()


# In[57]:


#Last 5 (default) lines viewed
dub_BikeDF.tail()


# In[58]:


#Describe function was used
print(dub_BikeDF['transaction_id'].describe())
print("----------")
print(dub_BikeDF['bike_type'].describe())
print("----------")
print(dub_BikeDF['dt_start'].describe())
print("----------")
print(dub_BikeDF['started_date'].describe())
print("----------")
print(dub_BikeDF['started_time'].describe())
print("----------")
print(dub_BikeDF['dt_end'].describe())
print("----------")
print(dub_BikeDF['ended_date'].describe())
print("----------")
print(dub_BikeDF['ended_time'].describe())
print("----------")


# In[59]:


#Unique function was used
print(dub_BikeDF['transaction_id'].unique())
print("----------")
print(dub_BikeDF['bike_type'].unique())
print("----------")
print(dub_BikeDF['dt_start'].unique())
print("----------")
print(dub_BikeDF['started_date'].unique())
print("----------")
print(dub_BikeDF['started_time'].unique())
print("----------")
print(dub_BikeDF['dt_end'].unique())
print("----------")
print(dub_BikeDF['ended_date'].unique())
print("----------")
print(dub_BikeDF['ended_time'].unique())
print("----------")


# ###   Views and Machine Learning

# In[60]:


sf_NumberOfBike = sf_BikeDF['rideable_type'].value_counts().reset_index() #Data arranged .
sf_NumberOfBike.columns = ['Type', 'Number'] #Coloumns are showed .
sf_NumberOfBike['Type'] = sf_NumberOfBike['Type'].replace('electric_bike', 'E_Bike') #Name is adjusted .
sf_NumberOfBike['Type'] = sf_NumberOfBike['Type'].replace('classic_bike', 'Classic_Bike') #Name is adjusted .


# In[61]:


#Pie visualization was done for San Francisco data
fig, ax = plt.subplots()
ax.pie(x=sf_NumberOfBike["Number"], labels=sf_NumberOfBike["Type"], autopct="%1.1f%%", 
       shadow=True, startangle=90, textprops={"size": "large"}, radius=3, 
       colors=["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"])
plt.show()


# This pie chart represents the distribution of three different bike types used in San Francisco, specifically E_Bike, Docked_Bike, and Classic_Bike. The proportions are displayed as percentages with the following key observations:
# 
# E_Bike occupies the largest share of the pie at 52.6%, indicating it is the most utilized type among the three.
# Docked_Bike makes up 30.8% of the total, showing a significant but secondary usage.
# Classic_Bike holds the smallest share at 16.6%, suggesting that this type is the least preferred or available among users.
# 

# In[62]:


#Pie visualization was done for San Francisco data
import matplotlib.pyplot as plt
E_Bike = 15995
Classic_Bike = 5164
durations = [E_Bike, Classic_Bike]
labels = ['E_Bike', 'Classic_Bike']
colors = ['blue', 'orange']
plt.figure(figsize=(8, 8))
plt.pie(durations, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('San Francisco Bike Usage Percentages ( Oct-Nov-Dec 2020) ')
plt.show()


# In[63]:


#Bar Chart visualization was done for San Francisco data
E_Bike = 15995
Classic_Bike = 5164

# Dataset
durations = [E_Bike, Classic_Bike]
labels = ['E_Bike', 'Classic_Bike']
colors = ['orange', 'green']

# Horizontal bar chart 
plt.figure(figsize=(10, 6))
plt.barh(labels, durations, color=colors)
plt.xlabel('Usage')
plt.ylabel('Categories')
plt.title('San Francisco Bike Usage ( Oct-Nov-Dec 2020)')
plt.show()


# In[64]:


dub_NumberOfBike = dub_BikeDF['bike_type'].value_counts().reset_index()
dub_NumberOfBike.columns = ['Type', 'Number']


# In[65]:


#Pie visualization was done for Dublin data
fig, ax = plt.subplots()
ax.pie(x=dub_NumberOfBike["Number"], 
       labels=dub_NumberOfBike["Type"], 
       autopct="%1.1f%%", 
       shadow=True, 
       startangle=90, 
       textprops={"size": "large"}, 
       radius=3, 
       colors=["#ff7f0e", "#1f77b4"]) # two colours are used
plt.show()


# This pie chart visualizes the distribution of two bike types in Dublin, categorized as Classic_Bike and E_Bike. 
# The vast majority of the chart, 97.6%, is occupied by Classic_Bike, shown in orange. This overwhelming share suggests that Classic_Bike is nearly the exclusive bike type in use or available in Dublin.
# Its dominance leaves very little room for any other bike type.
# E_Bike constitutes a mere 2.4% of the total, shown in blue. This small slice indicates that E_Bike has a negligible presence in the dataset compared to Classic_Bike.

# In[66]:


dictNumberOfBikeType = {"Type": ["San Francisco Bike", "San Francisco E-Bike", "Dublin Bike", "Dublin E-Bike"], 
                           "Value": [sf_NumberOfBike["Number"][0], sf_NumberOfBike["Number"][1], dub_NumberOfBike["Number"][0], dub_NumberOfBike["Number"][1]]}
numberOfBikeTypeDF = pd.DataFrame(dictNumberOfBikeType)


# In[67]:


#Barplot visualization was done
fig = plt.figure(figsize=(20,10))
sns.barplot(x="Type", y="Value", data=numberOfBikeTypeDF, alpha=0.9);


# This bar chart presents the distribution of bike usage two locations: San Francisco and Dublin, segmented by bike type (Bike and E-Bike). Here are the detailed insights:
# 
# The Dublin Bike category, shown in green, has an exceptionally high value, nearing 50,000. This is significantly larger than all other categories, indicating that Dublin has a predominant usage or availability of standard bikes compared to San Francisco.
# 
# For both cities, the E-Bike categories are minimal in comparison to the standard bike types:
# San Francisco E-Bike (orange) has a modest representation, contributing less than half the count of standard bikes in San Francisco.
# Dublin E-Bike (red) has a negligible value, almost imperceptible next to Dublin's standard bikes.
# 
# San Francisco's total bike count is substantially lower than Dublin’s, even when combining both bike types.
# The bar for San Francisco Bike (blue) is notable but still lags far behind Dublin’s standard bikes.
# 
# In San Francisco, the Bike category clearly dominates over the E-Bike category, with E-Bikes making up a smaller fraction of the total.
# In Dublin, the dominance of standard bikes is even more pronounced, with E-Bikes contributing a near-negligible proportion.
# 
# The overwhelming total bike count in Dublin compared to San Francisco raises questions about the underlying reasons—possibly due to differences in population, infrastructure, or cycling culture.
# This chart effectively highlights the stark differences in bike type usage between the two cities and offers a foundation for further analysis into the factors driving these patterns.

# ## Bike Usage Durations

# In[68]:


#Date and time columns converted to str format
sf_BikeDF['started_date'] = sf_BikeDF['started_date'].astype(str)
sf_BikeDF['started_time'] = sf_BikeDF['started_time'].astype(str)
sf_BikeDF['ended_date'] = sf_BikeDF['ended_date'].astype(str)
sf_BikeDF['ended_time'] = sf_BikeDF['ended_time'].astype(str)

#Converted to datetime format after combining date and time columns
sf_BikeDF['started_at'] = pd.to_datetime(sf_BikeDF['started_date'] + ' ' + sf_BikeDF['started_time'])
sf_BikeDF['ended_at'] = pd.to_datetime(sf_BikeDF['ended_date'] + ' ' + sf_BikeDF['ended_time'])

#Calculated time interval
sf_BikeDF['elapsed'] = sf_BikeDF['ended_at'] - sf_BikeDF['started_at']

#Set the time interval in seconds
sf_BikeDF['elapsed'] = sf_BikeDF['elapsed'].dt.total_seconds()

sf_BikeDF = sf_BikeDF[['ride_id', 'rideable_type', 'start_station_name', 'start_station_id', 'end_station_name', 'end_station_id', 'member_casual', 'started_at', 'ended_at', 'elapsed']]


# In[69]:


#When the update was made, the summation process was performed using the uniqe method to avoid repeating values. The reason for applying uniqe to seconds is to minimize the margin of error
sf_elapsedSum = sf_BikeDF['elapsed'].unique().sum()

sf_bikeSec = int(sf_elapsedSum)
sf_bikeMin = int(sf_elapsedSum/60)
sf_bikeHour = int(sf_bikeMin/60)


# In[70]:


#Date and time columns converted to str format
dub_BikeDF['started_date'] = dub_BikeDF['started_date'].astype(str)
dub_BikeDF['started_time'] = dub_BikeDF['started_time'].astype(str)
dub_BikeDF['ended_date'] = dub_BikeDF['ended_date'].astype(str)
dub_BikeDF['ended_time'] = dub_BikeDF['ended_time'].astype(str)

#Converted to datetime format after combining date and time columns
dub_BikeDF['started_at'] = pd.to_datetime(dub_BikeDF['started_date'] + ' ' + dub_BikeDF['started_time'])
dub_BikeDF['ended_at'] = pd.to_datetime(dub_BikeDF['ended_date'] + ' ' + dub_BikeDF['ended_time'])

#Calculated time interval
dub_BikeDF['elapsed'] = dub_BikeDF['ended_at'] - dub_BikeDF['started_at']

#Set the time interval in seconds
dub_BikeDF['elapsed'] = dub_BikeDF['elapsed'].dt.total_seconds()

dub_BikeDF = dub_BikeDF[['transaction_id', 'bike_type', 'started_at', 'ended_at', 'elapsed']]


# In[71]:


#When the update was made, the summation process was performed using the uniqe method to avoid repeating values. The reason for applying uniqe to seconds is to minimize the margin of error
dub_elapsedSum = dub_BikeDF['elapsed'].unique().sum()

dub_bikeSec = int(dub_elapsedSum)
dub_bikeMin = int(dub_elapsedSum/60)
dub_bikeHour = int(dub_bikeMin/60)


# In[72]:


#A new dataframe was created for the operations to be performed
timeData = {
    'Time': ["Second", "Minute", "Hour"],
    'San Francisco': [sf_bikeSec, sf_bikeMin, sf_bikeHour],
    'Dublin': [dub_bikeSec, dub_bikeMin, dub_bikeHour]
}
timeDF = pd.DataFrame(timeData)
#Data listed in dataframe
timeDF.style.background_gradient(axis=None, cmap="Blues")


# -------------------
"""
#The file to be processed is assigned to the variable
sf_BikeCsv102020": r"C:\Users\HP\Desktop\CA FILES\202010-baywheels-tripdata.csv
sf_BikeCsv112020": r"C:\Users\HP\Desktop\CA FILES\202011-baywheels-tripdata.csv
sf_BikeCsv122020": r"C:\Users\HP\Desktop\CA FILES\202012-baywheels-tripdata.csv
#All csv files read
sf_BikeCsv102020 = pd.read_csv(sf_BikeCsv102020)
sf_BikeCsv112020 = pd.read_csv(sf_BikeCsv112020)
sf_BikeCsv122020 = pd.read_csv(sf_BikeCsv122020)

sf_BikeCsv102020Rows = sf_BikeCsv102020.shape[0]
sf_BikeCsv112020Rows = sf_BikeCsv112020.shape[0]
sf_BikeCsv122020Rows = sf_BikeCsv122020.shape[0]

"""
# In[73]:


#Library is added
import pandas as pd

# File Paths
filePaths = {
    "sf_BikeCsv102020": r"C:\Users\HP\Desktop\CA FILES\202010-baywheels-tripdata.csv",
    "sf_BikeCsv112020": r"C:\Users\HP\Desktop\CA FILES\202011-baywheels-tripdata.csv",
    "sf_BikeCsv122020": r"C:\Users\HP\Desktop\CA FILES\202012-baywheels-tripdata.csv",
}

# Csv file are read and values are collected .
rowCounts = {}

for name, filePath in filePaths.items():
    try:
        newDF = pd.read_csv(filePath)
        rowCounts[name] = newDF.shape[0]
    except Exception as e:
        print(f"No read: {filePath}. Error: {e}")

# Numbers are assigned
sf_BikeCsv102020Rows = rowCounts.get("sf_BikeCsv102020", 0)
sf_BikeCsv112020Rows = rowCounts.get("sf_BikeCsv112020", 0)
sf_BikeCsv122020Rows = rowCounts.get("sf_BikeCsv122020", 0)

# Results are showed
print("2020-10 Row Number:", sf_BikeCsv102020Rows)
print("2020-11 Row Number::", sf_BikeCsv112020Rows)
print("2020-12 Row Number::", sf_BikeCsv122020Rows)

"""
#The file to be processed is assigned to the variable
dub_BikeCsv102020": r"C:\Users\HP\Desktop\CA FILES\moby-bikes-historical-data-102020.csv
dub_BikeCsv112020": r"C:\Users\HP\Desktop\CA FILES\moby-bikes-historical-data-112020.csv
dub_BikeCsv122020": r"C:\Users\HP\Desktop\CA FILES\moby-bikes-historical-data-122020.csv
#All csv files read
dub_BikeCsv102020 = pd.read_csv(dub_BikeCsv102020)
dub_BikeCsv112020 = pd.read_csv(dub_BikeCsv112020)
dub_BikeCsv122020 = pd.read_csv(dub_BikeCsv122020)

dub_BikeCsv102020Rows = dub_BikeCsv102020.shape[0]
dub_BikeCsv112020Rows = dub_BikeCsv112020.shape[0]
dub_BikeCsv122020Rows = dub_BikeCsv122020.shape[0]

"""
# In[74]:


#Library is added
import pandas as pd

# File Paths
filePaths = {
    "dub_BikeCsv102020": r"C:\Users\HP\Desktop\CA FILES\moby-bikes-historical-data-102020.csv",
    "dub_BikeCsv112020": r"C:\Users\HP\Desktop\CA FILES\moby-bikes-historical-data-112020.csv",
    "dub_BikeCsv122020": r"C:\Users\HP\Desktop\CA FILES\moby-bikes-historical-data-122020.csv",
}

# Csv file are read and values are collected 
rowCounts = {}

for name, filePath in filePaths.items():
    try:
        newDF = pd.read_csv(filePath)
        rowCounts[name] = newDF.shape[0]
    except Exception as e:
        print(f"No Read: {filePath}. Error: {e}")

# Numbers are assigned
dub_BikeCsv102020Rows = rowCounts.get("dub_BikeCsv102020", 0)
dub_BikeCsv112020Rows = rowCounts.get("dub_BikeCsv112020", 0)
dub_BikeCsv122020Rows = rowCounts.get("dub_BikeCsv122020", 0)

# Results are showed
print("2020-10 Row Number:", dub_BikeCsv102020Rows)
print("2020-11 Row Number:", dub_BikeCsv112020Rows)
print("2020-12 Row Number:", dub_BikeCsv122020Rows)


# In[75]:


dictMonthYearData = {"Location": ["San Francisco", "Dublin", "San Francisco", "Dublin", "San Francisco", "Dublin", ], 
                            "Month": ["Oct 2020", "Oct 2020", "Nov 2020", "Nov 2020", "Dec 2020", "Dec 2020"], 
                            "Value": [sf_BikeCsv102020Rows, dub_BikeCsv102020Rows, sf_BikeCsv112020Rows, dub_BikeCsv112020Rows, sf_BikeCsv122020Rows, dub_BikeCsv122020Rows]}
#Data listed in dataframe
monthYearData = pd.DataFrame(dictMonthYearData)
monthYearData[['Month', 'Year']] = monthYearData['Month'].str.split(' ', expand=True)
monthYearData['MonthYear'] = monthYearData['Month'] + ' ' + monthYearData['Year'].astype(str)
monthYearData.style.background_gradient(axis=0, gmap=monthYearData["Value"], cmap="RdYlBu")


# In[76]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Plotting
barWidth = 0.25
fig = plt.subplots(figsize =(12, 6))

# Adjusted the position of the bars on the X-axis
br1 = np.arange(len(monthYearData['Location'][monthYearData['Location'] == 'San Francisco'])) 
br2 = [x + barWidth for x in br1]

# Plot bars for San Francisco and Dublin
plt.bar(br1, monthYearData['Value'][monthYearData['Location'] == 'San Francisco'], color ='r', width = barWidth, edgecolor ='grey', label ='San Francisco') 
plt.bar(br2, monthYearData['Value'][monthYearData['Location'] == 'Dublin'], color ='g', width = barWidth, edgecolor ='grey', label ='Dublin')

# Set labels and title
plt.xlabel('Month', fontweight='bold', fontsize=12)
plt.ylabel('Total Usage', fontweight='bold', fontsize=12)
plt.xticks([r + barWidth for r in range(len(monthYearData['Location'][monthYearData['Location'] == 'San Francisco']))], monthYearData['MonthYear'][monthYearData['Location'] == 'San Francisco'])

# Show legend
plt.legend()
plt.show()


# This bar chart presents the distribution of bike usage two locations: San Francisco and Dublin .
# The Dublin Bike usage, shown in green, has an exceptionally high value, nearing 120,000. Only on December Dublin Bike usage is greater than San Francisco usage . Rest of months ,San Farncisco Bike usage , shown in red , has an exceptionally high value , minimum 130.000 and greater than Dublin Bike usage .
# For both cities min usage is nearing 100.000 . 
# On October there is a overwhelming difference between 2 cities . October is greatest usage for San Francisco .

# In[77]:


plt.figure(figsize=(12, 6))

# Line for San Francisco 
plt.plot(monthYearData['Month'][monthYearData['Location'] == 'San Francisco'], 
         monthYearData['Value'][monthYearData['Location'] == 'San Francisco'], 
         color='r', marker='o', label='San Francisco', linestyle='-', linewidth=2)

# Line for Dublin 
plt.plot(monthYearData['Month'][monthYearData['Location'] == 'Dublin'], 
         monthYearData['Value'][monthYearData['Location'] == 'Dublin'], 
         color='g', marker='s', label='Dublin', linestyle='-', linewidth=2)

# Headline and Labels are written
plt.title('Monthly Usage: San Francisco vs Dublin', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Usage', fontsize=12)
plt.xticks(rotation=45)  # Twisting on X line
plt.legend()

# Showing graph
plt.show()


# The plot offers a more detailed view compared to the bar chart, making it easier to discuss the maximum and minimum values and providing a clearer understanding of the numbers.

# In[78]:


# Values are made pivot
heatmap_data = monthYearData.pivot("Location", "Month", "Value")

# Map of temperature
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt="d", linewidths=1, linecolor='gray')

# Headline
plt.title('Monthly Usage Heatmap: San Francisco vs Dublin', fontsize=14, fontweight='bold')

# Showing plot
plt.show()


# In[79]:


plt.figure(figsize=(12, 6))

# Points are created for San Francisco and Dublin 
plt.scatter(monthYearData['Month'][monthYearData['Location'] == 'San Francisco'], 
            monthYearData['Value'][monthYearData['Location'] == 'San Francisco'], 
            color='r', label='San Francisco', s=100)

plt.scatter(monthYearData['Month'][monthYearData['Location'] == 'Dublin'], 
            monthYearData['Value'][monthYearData['Location'] == 'Dublin'], 
            color='g', label='Dublin', s=100)

# Headline and Labels are created
plt.title('Monthly Usage Scatter Plot: San Francisco vs Dublin', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Usage', fontsize=12)
plt.xticks(rotation=45)

# Legend Added
plt.legend()

# Showing Graph
plt.show()


# In[80]:


# Area Plot
plt.figure(figsize=(12, 6))

# Area plot is created to San Francisco and Dublin
plt.fill_between(monthYearData['Month'][monthYearData['Location'] == 'San Francisco'], 
                 monthYearData['Value'][monthYearData['Location'] == 'San Francisco'], 
                 color='r', alpha=0.5, label='San Francisco')
plt.fill_between(monthYearData['Month'][monthYearData['Location'] == 'Dublin'], 
                 monthYearData['Value'][monthYearData['Location'] == 'Dublin'], 
                 color='g', alpha=0.5, label='Dublin')

# Headline and Labels are created
plt.title('Monthly Usage Area Plot: San Francisco vs Dublin', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Usage', fontsize=12)
plt.xticks(rotation=45)

# Legend Added
plt.legend()

# Showing Graph
plt.show()


# In[81]:


#Median mean and variance printed to screen
median(monthYearData["Value"])
mean(monthYearData["Value"])
variance(monthYearData["Value"])


# In[82]:


monthYearData


# ## To enhance the analysis using machine learning techniques and generate more meaningful values, the upcoming months (January, February, March) will be forecasted using statistical methods. Specifically, we will employ methods such as mean, mode, and linear trend analysis to estimate the data for these months. By applying these statistical techniques, we aim to provide a broader set of values, allowing for more accurate predictions and insights. The ultimate goal is to improve the detection and classification of X and Y test values, which will help in refining the predictive model and enhancing its performance. These methods will facilitate a deeper understanding of the data, ensuring that future values are generated more precisely, which will contribute to better decision-making and model testing.

# In[83]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset
monthYearData = pd.DataFrame({
    'Location': ['San Francisco', 'Dublin', 'San Francisco', 'Dublin', 'San Francisco', 'Dublin'],
    'Month': ['Oct', 'Oct', 'Nov', 'Nov', 'Dec', 'Dec'],
    'Value': [167541, 122723, 133020, 117348, 106422, 123348],
    'Year': [2020, 2020, 2020, 2020, 2020, 2020],
    'MonthYear': ['Oct 2020', 'Oct 2020', 'Nov 2020', 'Nov 2020', 'Dec 2020', 'Dec 2020'],
})

# 1. (Mean) Calculation
mean_sf = monthYearData[monthYearData['Location'] == 'San Francisco']['Value'].mean()
mean_dublin = monthYearData[monthYearData['Location'] == 'Dublin']['Value'].mean()


# In[84]:


mean_sf , mean_dublin


# In[85]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset . January mean Dublin and Ssan Francisco Values are added 
monthYearData = pd.DataFrame({
    'Location': ['San Francisco', 'Dublin', 'San Francisco', 'Dublin', 'San Francisco', 'Dublin', 'San Francisco', 'Dublin' ],
    'Month': ['Oct', 'Oct', 'Nov', 'Nov', 'Dec', 'Dec' , ' Jan' , 'Jan'],
    'Value': [167541, 122723, 133020, 117348, 106422, 123348, 135661 , 121140 ],
    'Year': [2020, 2020, 2020, 2020, 2020, 2020 , 2021 , 2021 ],
    'MonthYear': ['Oct 2020', 'Oct 2020', 'Nov 2020', 'Nov 2020', 'Dec 2020', 'Dec 2020' , 'Jan 2021' , 'Jan 2021'],
})
# 2. (Median) Calculated
median_sf2021 = monthYearData[monthYearData['Location'] == 'San Francisco']['Value'].median()
median_dublin2021 = monthYearData[monthYearData['Location'] == 'Dublin']['Value'].median()


# In[86]:


median_sf2021 , median_dublin2021


# In[87]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset . February median Dublin and Ssan Francisco Values are added 
monthYearData = pd.DataFrame({
    'Location': ['San Francisco', 'Dublin', 'San Francisco', 'Dublin', 'San Francisco', 'Dublin', 'San Francisco', 'Dublin' , 'San Francisco', 'Dublin' ],
    'Month': ['Oct', 'Oct', 'Nov', 'Nov', 'Dec', 'Dec' , ' Jan' , 'Jan' , 'Feb' , 'Feb'],
    'Value': [167541, 122723, 133020, 117348, 106422, 123348, 135661 , 121140 , 134340 , 121931 ],
    'Year': [2020, 2020, 2020, 2020, 2020, 2020 , 2021 , 2021 , 2021 , 2021],
    'MonthYear': ['Oct 2020', 'Oct 2020', 'Nov 2020', 'Nov 2020', 'Dec 2020', 'Dec 2020' , 'Jan 2021' , 'Jan 2021' , 'Feb 2021 ' , 'Feb 2021 '],
})

# 3. Linear Trend Prediction
# Encode months to numerical values (Oct=1, Nov=2, Dec=3, etc.)
month_mapping = {'Oct': 1, 'Nov': 2, 'Dec': 3, 'Jan': 4, 'Feb': 5}
monthYearData['MonthNum'] = monthYearData['Month'].map(month_mapping)

# Linear Regression Model (San Francisco)
sf_data = monthYearData[monthYearData['Location'] == 'San Francisco']
X_sf = sf_data[['MonthNum']].values  # Bağımsız değişken (Ay numarası)
y_sf = sf_data['Value'].values  # Bağımlı değişken (Değer)

# NaN values are detected and wiped out
mask = ~np.isnan(X_sf).ravel() & ~np.isnan(y_sf)
X_sf = X_sf[mask]
y_sf = y_sf[mask]

sf_model = LinearRegression()
sf_model.fit(X_sf, y_sf)
oct_prediction_sf = sf_model.predict(np.array([[6]]))  # 6 is portreyed to March .

# Linear regression model (Dublin)
dublin_data = monthYearData[monthYearData['Location'] == 'Dublin']
X_dublin = dublin_data[['MonthNum']].values
y_dublin = dublin_data['Value'].values

# NaN values are detected and wiped out
mask = ~np.isnan(X_dublin).ravel() & ~np.isnan(y_dublin)
X_dublin = X_dublin[mask]
y_dublin = y_dublin[mask]

dublin_model = LinearRegression()
dublin_model.fit(X_dublin, y_dublin)
oct_prediction_dublin = dublin_model.predict(np.array([[6]]))  # 6 is portreyed to March .


# In[88]:


print("Linear Regression Predictions (March 2021):")
print(f"San Francisco: {oct_prediction_sf[0]:.0f}")
print(f"Dublin: {oct_prediction_dublin[0]:.0f}")


# In[89]:


# Dataset is fixed . March values are added 
monthYearData = pd.DataFrame({
    'Location': ['San Francisco', 'Dublin', 'San Francisco', 'Dublin', 'San Francisco', 'Dublin', 'San Francisco', 'Dublin' , 'San Francisco', 'Dublin' , 'San Francisco', 'Dublin' ],
    'Month': ['Oct', 'Oct', 'Nov', 'Nov', 'Dec', 'Dec' , ' Jan' , 'Jan' , 'Feb' , 'Feb' , 'Mar' , 'Mar'],
    'Value': [167541, 122723, 133020, 117348, 106422, 123348, 135661 , 121140 , 134340 , 121931 , 111525 , 121960 ],
    'Year': [2020, 2020, 2020, 2020, 2020, 2020 , 2021 , 2021 , 2021 , 2021, 2021 , 2021],
    'MonthYear': ['Oct 2020', 'Oct 2020', 'Nov 2020', 'Nov 2020', 'Dec 2020', 'Dec 2020' , 'Jan 2021' , 'Jan 2021' , 'Feb 2021 ' , 'Feb 2021 ' , 'Mar 2021', 'Mar 2021'],
})


# In[90]:


monthYearData #new dataset is displayed


# In[91]:


#Above Average Values has been created and values below the average are labeled as 0 and those above the average are labeled as 1
monthYearData["Above Average Values"] = 0
for i in range(6):
    if monthYearData["Value"][i] > np.mean(monthYearData["Value"]):
        monthYearData["Above Average Values"][i] = "1"
    else:
        monthYearData["Above Average Values"][i] = "0"
        
monthYearData


# In[92]:


#Bernoulli distribution
monthYearData["Above Average Values"].unique()
monthYearData["Above Average Values"] = monthYearData["Above Average Values"].astype(int)
sns.set_style("white")
fig,ax=plt.subplots(figsize=(12,8))
probabilities=monthYearData["Above Average Values"].value_counts(normalize=True)
ax=sns.barplot(x=probabilities.index, y=probabilities.values, palette="PuBuGn_r")
patches=ax.patches
label_graph(18,"Value","Probability","Bernoulli Distribution", 20)


# The result showing 80% '0' and 20% '1' in the plot likely reflects the distribution of the "Above Average Values" in your dataset.
# '0' (80%): This value represents the majority of the data. Since '0' corresponds to "below average" values (as implied by the name "Above Average Values"), it suggests that 80% of the observations in your dataset are below average based on whatever metric you're using to determine "above" or "below" average. This could mean that most of the data points fall below the threshold or reference value that determines whether a value is considered "above average."
# 
# '1' (20%): The value '1' represents "above average" values. Since 20% of the observations are above average, it suggests that a smaller portion of the data points meet the criteria for being above average. This might indicate that the threshold or the metric for "above average" is relatively high, and only a fifth of your data surpasses it.
# 
# This distribution might reflect a few things:
# Skewed Data: Your data is skewed towards below-average values, with a small proportion of the values meeting the "above average" criteria. This could be due to a variety of factors, such as the nature of the data, the way "average" is calculated, or the presence of extreme values (outliers) that are influencing the threshold.
# 
# Threshold Interpretation: The "average" value you're using to classify the data as above or below average could be relatively high compared to most of your data points. This suggests that most values fall below this threshold, and only a smaller percentage of the data exceed it.
# 
# Modeling or Classification Insight: If you’re using this distribution in a predictive model, it could help in understanding the imbalance between "above average" and "below average" values. For example, models might be biased toward predicting '0' (below average) more often due to the higher prevalence of this class. Techniques like resampling or weighting can be used to balance the impact of the two classes if necessary.
# 
# In summary, an 80% '0' and 20% '1' distribution means that the majority of your data points are below the average threshold you've set, while a smaller portion is above average. This is typical for datasets where the majority of values are centered around a lower range.

# In[93]:


#Binomial distribution
aboveAvarageValues=monthYearData[monthYearData["Above Average Values"]== 1]
aboveAvarageValues["Value"].value_counts(normalize=True)
n=12
p=0.28
x=np.arange(0,2)
fig,ax=plt.subplots(figsize=(12,8))
pmf=stats.binom.pmf(x,n,p)
pps=plt.bar(x,pmf)
print(pmf)
plt.locator_params(integer=True)
label_graph(15,"Above Average Values", "Probability", "Binomial Distribution",20)


# 
# A binomial distribution is used to model situations where there are two possible outcomes (e.g., success or failure) in a fixed number of trials. Each trial is independent, and the probability of success is the same for each trial. It is defined by:
# 
# n = the number of trials
# p = the probability of success on a single trial
# (1 - p) = the probability of failure on a single trial
# Interpreting 80% '1' and 20% '0'
# In your case:
# 
# 80% '1': This represents the "success" outcome, meaning that 80% of the data points fall into the "success" category. It indicates that in most cases, the event being measured is successful.
# 20% '0': This represents the "failure" outcome, meaning that 20% of the data points fall into the "failure" category. This suggests that in a smaller proportion of cases, the event does not succeed.
# 
# Probability of Success (p): The success rate (represented by '1') is 80%, meaning that the event has an 80% chance of being successful.
# Probability of Failure (1 - p): The failure rate (represented by '0') is 20%, meaning there is a 20% chance that the event will not be successful.
# In the context of a binomial distribution, you would expect 80% of the trials to be successful and 20% to fail, assuming the probability of success remains constant for each trial.
# 80% '1' and 20% '0' means that the probability of success is high (80%), and the probability of failure is low (20%).
# The distribution is binomial because there are two possible outcomes (success or failure), and it follows a fixed probability of success across trials.

# ----------

# In[94]:


monthYearData = monthYearData.drop(columns='Above Average Values')


# In[95]:


monthYearData


# In[96]:


monthYearData.dtypes


# In[97]:


#Label encoder was applied to the selected columns
columnLabelEncode = ["Location", "Month", "Year", "MonthYear"]
labelEncoder = LabelEncoder()
for column in columnLabelEncode:
    monthYearData[column] = labelEncoder.fit_transform(monthYearData[column])


# In[98]:


monthYearData.dtypes


# In[99]:


#After the label encoder was made, the describe function was applied to the dataframe
monthYearData.describe()


# In[100]:


#For processing the columns were divided into x and y
X = monthYearData.drop("Location", axis = 1)
print(X)
y = monthYearData.Year.values.astype(int)
print(y)


# In[101]:


#Each column was standardized and divided into two as train and test
X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)
print(y_train.mean())
print(y_test.mean())


# The result showing 0.5 for both y_train.mean() and y_test.mean() means that the average (mean) value of the target variable y is 50% for both the training and testing sets. 
# 
# y_train.mean() = 0.5: This indicates that in the training dataset (X_train and y_train), 50% of the target values (y_train) are 1, and the remaining 50% are 0. Essentially, the target variable is evenly split between the two classes (if it's a binary classification problem).
# 
# y_test.mean() = 0.5: Similarly, in the test dataset (X_test and y_test), 50% of the target values (y_test) are 1, and 50% are 0. The test set has the same class distribution as the training set.
# 
# The fact that the mean is 0.5 suggests that your dataset is balanced. This means that there is an equal number of instances of both classes (0 and 1) in your target variable y. This is a good thing in many cases because imbalanced datasets (where one class is much more frequent than the other) can cause models to be biased towards predicting the more common class.
# 
# The target variable is binary (0 and 1), and you get a mean of 0.5, it suggests that your dataset is perfectly balanced with an equal distribution of the two classes.

# In[102]:


#SVC object created. C = 1 received. Model trained and prediction made
model = SVC(C = 1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[103]:


#Confusion matrix was used
metrics.confusion_matrix(y_true=y_test, y_pred=y_pred)


# In[104]:


#Accuracy, Precision Score, Recall Score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision Score:",precision_score(y_test, y_pred, pos_label="positive", average="micro"))
print("Recall Score:",recall_score(y_test, y_pred, pos_label="positive", average="micro"))


# In[105]:


#Created a KFold object with 5 splits 
folds = KFold(n_splits = 5, shuffle = True, random_state = 4)
model = SVC(C = 1)


# In[106]:


#Computed the cross-validation scores. Printed 5 accuracies obtained from the 5 folds
cv_results = cross_val_score(model, X_train, y_train, cv = folds, scoring = "accuracy") 
print(cv_results)
print("Mean Accuracy = {}".format(cv_results.mean()))


# In[107]:


#Specified range of parameters (C) as a list and set up grid search scheme
params = {"C": [0.1, 1, 10, 100, 1000]}
model = SVC()
model_cv = GridSearchCV(estimator = model, param_grid = params, scoring= "accuracy", cv = folds, verbose = 1, return_train_score=True)      


# In[108]:


#Model trained and results printed on screen
model_cv.fit(X_train, y_train)  
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[109]:


#Plot of C versus train and test scores
plt.figure(figsize=(8, 6))
plt.plot(cv_results["param_C"], cv_results["mean_test_score"])
plt.plot(cv_results["param_C"], cv_results["mean_train_score"])
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend(["Test Accuracy", "Train Accuracy"], loc="upper left")
plt.xscale("log")


# In[110]:


#The highest test accuracy printed on the screen
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']
print(" The highest test accuracy is {0} at C = {1}".format(best_score, best_C))


# In[111]:


#The model with the best C value was trained and made predictions
model = SVC(C=best_C)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[112]:


#Accuracy, Precision Score, Recall Score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision Score:",precision_score(y_test, y_pred, pos_label="positive", average="micro"))
print("Recall Score:",recall_score(y_test, y_pred, pos_label="positive", average="micro"))


# In[113]:


#Standard scaler object created.
scalar = StandardScaler()
scalar.fit(monthYearData)
scaled_data = scalar.transform(monthYearData)
scaled_data


# In[114]:


#PCA = 2 received
pca = PCA(n_components = 2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
x_pca.shape


# In[115]:


#Visualized with scatter
plt.figure(figsize = (8, 6))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c = monthYearData["Location"], cmap ="plasma")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")


# In[116]:


monthYearData


# In[117]:


#The algorithms were enabled to predict the entered data 
x = monthYearData.iloc[:,[1,2,3,4]]
y = monthYearData.iloc[:,0]
X = x.values
Y = y.values
enterData = [1,111525,1,3] # ->1


# In[118]:


#Data split for training and testing. It was later scaled
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
X_train = scalar.fit_transform(x_train)
X_test = scalar.transform(x_test)


# In[119]:


#Logistic Regression Classifier and Confusion Matrix
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test) 
cm = confusion_matrix(y_test,y_pred)
print("Logistic Regression Classifier")
print("Forecasting with data entry:",logr.predict([enterData]))
print(cm)
accuracyLOG = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:")
print(accuracyLOG)


# In[120]:


#K-Nearest Neighbors Classifier (KNN) and Confusion Matrix
knn = KNeighborsClassifier(n_neighbors=1, metric="minkowski")
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("K-Nearest Neighbors Classifier")
print("Forecasting with data entry:",knn.predict([enterData]))
print(cm)
accuracyKNN = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:")
print(accuracyKNN)


# In[121]:


#Support Vector Machine Classifier (SVC) and Confusion Matrix
svc = SVC(kernel="poly")
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Support Vector Machine Classifier")
print("Forecasting with data entry:",svc.predict([enterData]))
print(cm)
accuracySVC = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:")
print(accuracySVC)


# In[122]:


#Naive Bayes Classifier and Confusion Matrix
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Naive Bayes Classifier")
print("Forecasting with data entry:",gnb.predict([enterData]))
print(cm)
accuracyNB = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:")
print(accuracyNB)


# In[123]:


#Decision Tree Classifier and Confusion Matrix
dtc = DecisionTreeClassifier(criterion = "entropy")
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Decision Tree Classifier")
print("Forecasting with data entry:",dtc.predict([enterData]))
print(cm)
accuracyDT = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:")
print(accuracyDT)


# In[124]:


#Random Forest Classifier and Confusion Matrix
rfc = RandomForestClassifier(n_estimators=10, criterion = "entropy")
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Random Forest Classifier")
print("Forecasting with data entry:",rfc.predict([enterData]))
print(cm)
accuracyRF = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:")
print(accuracyRF)


# In[125]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define model results
models = [
    'KNeighborsRegressor',
    'Support Vector Machine Classifier',
    'RandomForestRegressor',
    'DecisionTreeRegressor',
    'Naive Bayes Classifier ' ,
    'Logistic Regression Classifier'
]

accurancy = [
    0.5,
    0.25,
    0.5,
    0.5,
    1 ,
    0.5
]

# Create DataFrame
results_df = pd.DataFrame({
    'Model': models,
    'Forecast': accurancy
})

# Create plot
fig, ax = plt.subplots(figsize=(14, 8))  # Fixed subplot configuration, (1, figsize) is incorrect

# Forecast Plot
sns.barplot(x='Forecast', y='Model', data=results_df, ax=ax, palette='viridis')  # ax[1] is incorrect; should be ax
ax.set_title('Next Month Forecasts')  # Updated ax[1] to ax
ax.set_xlabel('Accurancy')  # Updated ax[1] to ax
ax.set_ylabel('Model')  # Updated ax[1] to ax

# Adjust layout
plt.tight_layout()
plt.show()


# The accuracy rates are mainly 0.25, 0.5, and 1.0, which raises some interesting points.
# Naive Bayes Classifier has achieved 100% accuracy, which is unusual and may indicate potential issues like data leakage or overfitting. Dataset was scaled by StandardScaler before .
# Support Vector Machine Classifier (SVM) has the lowest accuracy of 25%, suggesting the model might require better data preprocessing, scaling, or hyperparameter tuning.

# In[ ]:




