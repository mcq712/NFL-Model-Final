# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:25:03 2022

@author: micha
"""

import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import seaborn as sns
# from urllib.request import urlopen
# from joblib.numpy_pickle_utils import xrange
# from lxml.html import fromstring
# import requests
# display entire table
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
pd.set_option('display.max_rows', None, 'display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

dvoa = pd.read_csv(r"C:\Users\micha\OneDrive\Desktop\nfl\scraped_data\dvoa\dvoa_2021-2022.csv")
dvoa.Year = dvoa.Year.astype(str)
dvoa.Week = dvoa.Week.astype(str)


team_abbrev_match = pd.read_csv(r'C:\Users\micha\OneDrive\Desktop\nfl\misc\abbreviations_match.csv')
dvoa_final = dvoa.merge(team_abbrev_match, on='Team2')

dvoa_final.Year = dvoa_final.Year.astype(str)

# # dvoa_final.Tm_Score = dvoa_final.Tm_Score.astype(int)
# # dvoa_final.ORtg = dvoa_final.ORtg.astype(float)
# # dvoa_final.DRtg = dvoa_final.DRtg.astype(float)
# # dvoa_final.Pace = dvoa_final.Pace.astype(float)
# # dvoa_final.Ftr = dvoa_final.Ftr.astype(float)
# # dvoa_final.ThreeAr = dvoa_final.ThreeAr.astype(float)
# # dvoa_final.TSpct = dvoa_final.TSpct.astype(float)
# # dvoa_final.TRBpct = dvoa_final.TRBpct.astype(float)
# # dvoa_final.ASTpct = dvoa_final.ASTpct.astype(float)
# # dvoa_final.STLpct = dvoa_final.STLpct.astype(float)
# ## Create the Game ID and cast values as floats

dvoa_final.insert(0, 'GameID', dvoa_final['Team']+dvoa_final['Week']+dvoa_final['Year'])

# Call URL
# team = input('Please use team abbreviation: ')
team_abbreviations = ['buf', 'mia', 'nwe', 'nyj', 'htx', 'clt', 'jax', 'oti', 'cin', 'pit',
                      'cle', 'rav', 'den', 'kan', 'rai', 'sdg', 'phi', 'dal', 'nyg', 'was',
                      'car', 'nor', 'tam', 'atl', 'chi', 'det', 'gnb', 'min', 'crd', 'ram', 'sea', 'sfo']
df_holder = []
df_holder_2021 = []
for i in team_abbreviations:
    url_main = 'https://www.pro-football-reference.com/teams/{0}/2021/gamelog/'.format(i)
    data = pd.read_html(url_main)[0]
    data.rename(columns={'Unnamed: 3_level_1': 'Box_link'}, inplace=True)
    data.rename(columns={'Unnamed: 4_level_1': 'W/L'}, inplace=True)
    data.rename(columns={'Unnamed: 6_level_1': '@'}, inplace=True)
    data.rename(columns={'Unnamed: 3_level_1': 'Box_link'}, inplace=True)
    data.columns = ['Week', 'Day', 'Date', 'Box_link', 'W/L', 'OT', 'Home/Away', 'Opp_Team', 'Tm_Score',
                    'Opp_Score', 'PassCmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'Sk', 'YdsLost_Sk', 'PassY/A',
                    'PassNY/A', 'Cmp%', 'PasserRate', 'RushAtt', 'RushYds', 'RushY/A', 'RushTD', 'FGM', 'FGA', 'XPM',
                    'XPA', 'Pnt', 'PntYds', 'ThirdConv', '3DAtt', '4DConv', '4DAtt', 'ToP']
    data = data[pd.notnull(data['Opp_Score'])]
    data.insert(0, 'Team', i)
    data.insert(11, 'Total Score', data['Tm_Score'] + data['Opp_Score'])
    df_holder.append(data)
for i in team_abbreviations:
    url_main_2021 = 'https://www.pro-football-reference.com/teams/{0}/2022/gamelog/'.format(i)
    data_extra = pd.read_html(url_main_2021)[0]
    data_extra.rename(columns={'Unnamed: 3_level_1': 'Box_link'}, inplace=True)
    data_extra.rename(columns={'Unnamed: 4_level_1': 'W/L'}, inplace=True)
    data_extra.rename(columns={'Unnamed: 6_level_1': '@'}, inplace=True)
    data_extra.rename(columns={'Unnamed: 3_level_1': 'Box_link'}, inplace=True)
    data_extra.columns = ['Week', 'Day', 'Date', 'Box_link', 'W/L', 'OT', 'Home/Away', 'Opp_Team', 'Tm_Score',
                          'Opp_Score', 'PassCmp', 'PassAtt', 'PassYds', 'PassTD', 'Int', 'Sk', 'YdsLost_Sk', 'PassY/A',
                          'PassNY/A', 'Cmp%', 'PasserRate', 'RushAtt', 'RushYds', 'RushY/A', 'RushTD', 'FGM', 'FGA',
                          'XPM',
                          'XPA', 'Pnt', 'PntYds', 'ThirdConv', '3DAtt', '4DConv', '4DAtt', 'ToP']
    data_extra = data_extra[pd.notnull(data_extra['Opp_Score'])]
    data_extra.insert(0, 'Team', i)
    data_extra.insert(11, 'Total Score', data_extra['Tm_Score'] + data_extra['Opp_Score'])
    df_holder_2021.append(data_extra)
df_2021 = pd.concat(df_holder)
df_2021.insert(1,'Year', '2021')
df_2021.Week = df_2021.Week.astype(str)
df_2021.Year = df_2021.Year.astype(str)
df_2022 = pd.concat(df_holder_2021)

df_2022.Week = df_2022.Week.astype(str)
df_2022.insert(1,'Year', '2022')
df_2022.Year = df_2022.Year.astype(str)

all_dfs = pd.concat([df_2021, df_2022], axis=0)
all_dfs.insert(0, 'GameID', all_dfs['Team']+all_dfs['Week']+all_dfs['Year'])
all_dfs.Week = all_dfs.Week.astype(int)


# 	
# # Drop a row by condition


all_dfs = all_dfs[all_dfs.Week>=2]

final_df=all_dfs.merge(dvoa_final, on='GameID')

final_df.to_csv(r'C:\Users\micha\OneDrive\Desktop\nfl\half_cleaned_data\half_cleaned_week8.csv')

clean_df = pd.read_csv(r"C:\Users\micha\OneDrive\Desktop\nfl\full_cleaned_data\full_cleaned_week8.csv")



### Model

clean_df.insert(8,'Margin', clean_df['Tm_Score']-clean_df['Opp_Score'])

# clean_df.Offense_Weighted_DVOA = clean_df.Offense_Weighted_DVOA.astype(float)
# clean_df.Defense_Weighted_DVOA = clean_df.Defense_Weighted_DVOA.astype(float)
# clean_df.Away_Offense_Weighted_DVOA = clean_df.Away_Offense_Weighted_DVOA.astype(float)
# clean_df.Away_Defense_Weighted_DVOA = clean_df.Away_Defense_Weighted_DVOA.astype(float)

X = clean_df[['Offense_Weighted_DVOA', 'Defense_Weighted_DVOA', 
              'Away_Offense_Weighted_DVOA', 'Away_Defense_Weighted_DVOA']]
y = clean_df['Margin']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.40,random_state=50)

rf = RandomForestRegressor(n_estimators = 10000)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

def plotGraph(y_test,y_pred,regressorName):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.scatter(range(len(y_test)), y_test, color='blue')
    plt.scatter(range(len(y_pred)), y_pred, color='red')
    plt.title(regressorName)
    plt.show()
    return




plotGraph(y_test,y_pred,'Test')
# print('Accuracy: ',metrics.accuracy_score(y_test, y_pred))

#Calculate the predictions
#Calculate the root mean squared error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(rmse)



#use the forest's predict method on the test data
# Calc absolute erros
errors = abs(y_pred - y_test)
mape = 100 * (errors / y_test)

#Print out the mae
print('Mean Absolute Error', round(np.mean(errors), 2), 'points.')
accuracy = 100 - np.mean(mape)
print('Accuracy: ', round(accuracy, 2), '%')
feature_list = list(clean_df[['Offense_Weighted_DVOA', 'Defense_Weighted_DVOA', 
                        'Away_Offense_Weighted_DVOA', 'Away_Defense_Weighted_DVOA']])

importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 10)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


dvoa_ratings = pd.read_csv(r"C:\Users\micha\OneDrive\Desktop\nfl\scraped_data\dvoa\dvoa_after_7.csv")

dvoa_ratings.Offense_Weighted_DVOA = dvoa_ratings.Offense_Weighted_DVOA.astype(float)
dvoa_ratings.Defense_Weighted_DVOA = dvoa_ratings.Defense_Weighted_DVOA.astype(float)

o_weight_dvoa = dvoa_ratings.groupby('Team').Offense_Weighted_DVOA.mean()
d_weight_dvoa = dvoa_ratings.groupby('Team').Defense_Weighted_DVOA.mean()
##########
#########
# TKINTER
import tkinter as tk 


root= tk.Tk()
root.title("Random Forest Using DVOA -- Predicting Winning Team and Margin")


canvas1 = tk.Canvas(root, width = 500, height = 350)
canvas1.pack()

label1 = tk.Label(root, text='Home:  ')
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root)
canvas1.create_window(270, 100, window=entry1)


label2 = tk.Label(root, text='Away: ')
canvas1.create_window(100, 120, window=label2)

entry2 = tk.Entry (root)
canvas1.create_window(270, 120, window=entry2)



def values(): 
    global Offense_Weighted_DVOA
    Offense_Weighted_DVOA = o_weight_dvoa[entry1.get()]
    
    global Defense_Weighted_DVOA
    Defense_Weighted_DVOA = d_weight_dvoa[entry1.get()]
    
    global Away_Offense_Weighted_DVOA
    Away_Offense_Weighted_DVOA = o_weight_dvoa[entry2.get()]
    
    global Away_Defense_Weighted_DVOA
    Away_Defense_Weighted_DVOA = d_weight_dvoa[entry2.get()]

    
    Prediction_result  = ('  Predicted spread: ', rf.predict([[Offense_Weighted_DVOA,Defense_Weighted_DVOA,Away_Offense_Weighted_DVOA,Away_Defense_Weighted_DVOA]]))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='sky blue')
    canvas1.create_window(270, 280, window=label_Prediction)
    
button1 = tk.Button (root, text='      Predict spread      ',command=values, bg='red', fg='white', font=11)
canvas1.create_window(270, 220, window=button1)
 
root.mainloop()

