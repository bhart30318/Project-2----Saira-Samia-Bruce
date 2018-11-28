# Preface
We first used an open dataset from the popular site Kaggle. This European Soccer Database (database1.sqlite) has more than 25,000 matches and more than 10,000 players for European professional soccer seasons from 2008 to 2016.  The second dataset (teams.csv) we used is also from Kaggle and includes information on all European professional soccer teams, including the Country name and three-letter Country code. 

# Dataset

The goal of this notebook is to provide the process that we undertook to gather the two datasets, importing Python libraries for the analysis, reading the datasets, exploring the database attributes, cleaning data (including removing null data), performing a feature correlation to attributes in the sqlite database, visualizing the output of the feature correlation, analyzing the findings, merging this dataset with the .csv dataset, analyzing the tabular outputs of this merge, and finally, creating a .sqlite engine.  

# Getting Started

To get started, we:

1. Download the data from: https://www.kaggle.com/hugomathien/soccer
2. Extract the zip file called "soccer.zip"
3. The extracted file includes the .sqlite database
4. Separately, from 1., we reviewed the .txt files for all country soccer teams. Used the European soccer team .txt file to:
	a. create the .csv for review and merge with the database file from 3.

# Import Libraries
We will start by importing the Python libraries we will be using in this analysis. These libraries include: sqllite3 for interacting with a local relational database pandas and numpy for data ingestion and manipulation matplotlib for data visualization specific methods from sklearn for Machine Learning and customplot, which contains custom functions we have written for this notebook

	import sqlite3
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.cluster import KMeans
	from sklearn.preprocessing import scale
	from customplot import *
	from sqlalchemy import create_engine

# Ingest Data
Now, we will need to read the dataset using the commands below.

Created connection.

	database = "database1.sqlite"
	conn = sqlite3.connect(database)
	df = pd.read_sql_query("SELECT * FROM Player_Attributes", conn)

	
# Exploring Data
We started our data exploration by generating simple statistics of the data. 

	df.columns
	Index(['id', 'player_fifa_api_id', 'player_api_id', 'date', 'overall_rating',
	       'potential', 'preferred_foot', 'attacking_work_rate',
	       'defensive_work_rate', 'crossing', 'finishing', 'heading_accuracy',
	       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
	       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
	       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
	       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
	       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
	       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
	       'gk_reflexes'],
	      dtype='object')

	  df.describe().transpose()	    

# Data Cleaning: Handling Missing Data

is any row NULL ?

	df.isnull().any().any(), df.shape
	(True, (183978, 42))

Now let’s try to find how many data points in each column are null.

	df.isnull().sum(axis=0)
	id                        0
	player_fifa_api_id        0
	player_api_id             0
	date                      0
	overall_rating          836
	potential               836
	preferred_foot          836
	attacking_work_rate    3230
	defensive_work_rate     836
	crossing                836
	finishing               836
	heading_accuracy        836
	short_passing           836
	volleys                2713
	dribbling               836
	curve                  2713
	free_kick_accuracy      836
	long_passing            836
	ball_control            836
	acceleration            836
	sprint_speed            836
	agility                2713
	reactions               836
	balance                2713
	shot_power              836
	jumping                2713
	stamina                 836
	strength                836
	long_shots              836
	aggression              836
	interceptions           836
	positioning             836
	vision                 2713
	penalties               836
	marking                 836
	standing_tackle         836
	sliding_tackle         2713
	gk_diving               836
	gk_handling             836
	gk_kicking              836
	gk_positioning          836
	gk_reflexes             836
	dtype: int64

# Fixing Null Values by Deleting Them
In our next two lines, we dropped the null values by going through each row.

Take initial # of rows

	rows = df.shape[0]

Drop the NULL rows

	df = df.dropna()

Now if we check the null values and number of rows, we will see that there are no null values and number of rows decreased accordingly.

Check if all NULLS are gone ?

	print(rows)
	df.isnull().any().any(), df.shape
	183978

	(False, (180354, 42))

3624

Our data table has many lines so we can only look at few lines at once. Instead of looking at same top 10 lines every time, we shuffle to get a distributed sample when we display top few rows

	df = df.reindex(np.random.permutation(df.index))

Let’s take a look at top few rows.

We used the head function for data frames for this task. This gives us every column in every row.

	df.head(5)


 <table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>player_fifa_api_id</th>
      <th>player_api_id</th>
      <th>date</th>
      <th>overall_rating</th>
      <th>potential</th>
      <th>preferred_foot</th>
      <th>attacking_work_rate</th>
      <th>defensive_work_rate</th>
      <th>crossing</th>
      <th>...</th>
      <th>vision</th>
      <th>penalties</th>
      <th>marking</th>
      <th>standing_tackle</th>
      <th>sliding_tackle</th>
      <th>gk_diving</th>
      <th>gk_handling</th>
      <th>gk_kicking</th>
      <th>gk_positioning</th>
      <th>gk_reflexes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>111083</th>
      <td>111084</td>
      <td>172835</td>
      <td>181984</td>
      <td>2013-03-08 00:00:00</td>
      <td>77.0</td>
      <td>80.0</td>
      <td>right</td>
      <td>low</td>
      <td>high</td>
      <td>39.0</td>
      <td>...</td>
      <td>38.0</td>
      <td>49.0</td>
      <td>80.0</td>
      <td>84.0</td>
      <td>76.0</td>
      <td>12.0</td>
      <td>11.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>129006</th>
      <td>129007</td>
      <td>186362</td>
      <td>119118</td>
      <td>2008-08-30 00:00:00</td>
      <td>63.0</td>
      <td>65.0</td>
      <td>right</td>
      <td>medium</td>
      <td>medium</td>
      <td>31.0</td>
      <td>...</td>
      <td>46.0</td>
      <td>62.0</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>25.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>12.0</td>
      <td>7.0</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>45162</th>
      <td>45163</td>
      <td>213135</td>
      <td>426202</td>
      <td>2013-02-15 00:00:00</td>
      <td>58.0</td>
      <td>75.0</td>
      <td>right</td>
      <td>high</td>
      <td>medium</td>
      <td>48.0</td>
      <td>...</td>
      <td>46.0</td>
      <td>54.0</td>
      <td>29.0</td>
      <td>23.0</td>
      <td>17.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>15.0</td>
      <td>10.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>79520</th>
      <td>79521</td>
      <td>205402</td>
      <td>402975</td>
      <td>2007-02-22 00:00:00</td>
      <td>58.0</td>
      <td>76.0</td>
      <td>right</td>
      <td>medium</td>
      <td>medium</td>
      <td>52.0</td>
      <td>...</td>
      <td>53.0</td>
      <td>60.0</td>
      <td>29.0</td>
      <td>25.0</td>
      <td>24.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>8.0</td>
      <td>10.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>20970</th>
      <td>20971</td>
      <td>177635</td>
      <td>41092</td>
      <td>2013-02-15 00:00:00</td>
      <td>78.0</td>
      <td>82.0</td>
      <td>left</td>
      <td>high</td>
      <td>medium</td>
      <td>86.0</td>
      <td>...</td>
      <td>78.0</td>
      <td>76.0</td>
      <td>74.0</td>
      <td>75.0</td>
      <td>79.0</td>
      <td>5.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>14.0</td>
      <td>8.0</td>
    </tr>
  </tbody>
</table>

5 rows × 42 columns

How does a player's overall rating compare to their number of penalties?

	df[:10][['penalties', 'overall_rating']]

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>penalties</th>
      <th>overall_rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>48.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>48.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48.0</td>
      <td>62.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>47.0</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>47.0</td>
      <td>61.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>59.0</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>59.0</td>
      <td>74.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>59.0</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>59.0</td>
      <td>73.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>59.0</td>
      <td>73.0</td>
    </tr>
  </tbody>
</table>

# Feature Correlation Analysis
Next, we will check if ‘penalties’ is correlated to ‘overall_rating’. Are these correlated (using Pearson’s correlation coefficient) ?

	df['overall_rating'].corr(df['penalties'])

0.39271510791118897

We see that Pearson’s Correlation Coefficient for these two columns is 0.39. 

Pearson's coefficient has a range from -1 to +1. A value of 0 would have told there is no correlation, so we shouldn’t bother looking at that attribute. A value of 0.39 shows some correlation, although it could be stronger. We have these attributes which are slightly correlated. 

Next, we will create a list of features that we would like to iterate the same operation on.

Create a list of potential Features that you want to measure correlation with

	potentialFeatures = ['acceleration', 'curve', 'free_kick_accuracy', 'ball_control', 'shot_power', 'stamina']

The for loop below prints out the correlation coefficient of “overall_rating” of a player with each feature we added to the list as potential.

check how the features are correlated with the overall ratings

	for f in potentialFeatures:
    	related = df['overall_rating'].corr(df[f])
    	print("%s: %f" % (f,related))

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>acceleration: 0.243998
curve: 0.357566
free_kick_accuracy: 0.349800
ball_control: 0.443991
shot_power: 0.428053
stamina: 0.325606
</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

Which features have the highest correlation with overall_rating?

Looking at the values printed by the previous cell, we noticed that the to two are “ball_control” (0.44) and “shot_power” (0.43). So these two features seem to have higher correlation with “overall_rating”.

# Data Visualization:
Next we plotted the correlation coefficients of each feature with “overall_rating”. We started by selecting the columns and creating a list with correlation coefficients, called “correlations”.

	cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',
	       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
	       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
	       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
	       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
	       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
	       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
	       'gk_reflexes']

We created a list containing Pearson's correlation between 'overall_rating' with each column in cols

	correlations = [ df['overall_rating'].corr(df[f]) for f in cols ]
	len(cols), len(correlations)
	(34, 34)

We make sure that the number of selected features and the correlations calculated are the same, e.g., both 34 in this case. 

	def plot_dataframe(df, y_label):  
    color='coral'
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.ylabel(y_label)

```markdown

    ax = df2.correlation.plot(linewidth=3.3, color=color)
    ax.set_xticks(df2.index)
    ax.set_xticklabels(df2.attributes, rotation=75)
    plt.show()
    plt.savefig("playerstats.png")
```

	df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations}) 

We plotted this dataframe using the function we created

```markdown    
	plot_dataframe(df2, 'Player\'s Overall Rating')
```
<p>
	<img src="playerstats.png" alt="png">
</p>

# Analysis of Findings

	select5features = ['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']
	select5features

['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']

	['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']

['gk_kicking', 'potential', 'marking', 'interceptions', 'standing_tackle']

	df_select = df[select5features].copy(deep=True)
	df_select.head()

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gk_kicking</th>
      <th>potential</th>
      <th>marking</th>
      <th>interceptions</th>
      <th>standing_tackle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>71.0</td>
      <td>65.0</td>
      <td>70.0</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0</td>
      <td>71.0</td>
      <td>65.0</td>
      <td>70.0</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.0</td>
      <td>66.0</td>
      <td>65.0</td>
      <td>41.0</td>
      <td>66.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.0</td>
      <td>65.0</td>
      <td>62.0</td>
      <td>40.0</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.0</td>
      <td>65.0</td>
      <td>62.0</td>
      <td>40.0</td>
      <td>63.0</td>
    </tr>
  </tbody>
</table>

	data = scale(df_select)

	#Define number of clusters
	noOfClusters = 4

	#Train a model
	model = KMeans(init='k-means++', n_clusters=noOfClusters, n_init=20).fit(data)

	print(90*'_')
	print("\nCount of players in each cluster")
	print(90*'_')

	pd.value_counts(model.labels_, sort=False)

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt"></div><div class="output_subarea output_text output_stream output_stdout"><pre>__________________________________________________________________________________________

Count of players in each cluster
__________________________________________________________________________________________
</pre></div></div><div class="output_area"><div class="prompt output_prompt"><bdi>Out[42]:</bdi></div><div class="output_subarea output_text output_result"><pre>0    50490
1    55903
2    23777
3    50184
dtype: int64</pre></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

# Tables

	tables = pd.read_sql("SELECT * FROM sqlite_master where type='table'" ,conn)

	tables.shape

(8, 5)

	tables

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[45]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>name</th>
      <th>tbl_name</th>
      <th>rootpage</th>
      <th>sql</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>table</td>
      <td>sqlite_sequence</td>
      <td>sqlite_sequence</td>
      <td>4</td>
      <td>CREATE TABLE sqlite_sequence(name,seq)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>table</td>
      <td>Player_Attributes</td>
      <td>Player_Attributes</td>
      <td>11</td>
      <td>CREATE TABLE "Player_Attributes" (\n\t`id`\tIN...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>table</td>
      <td>Player</td>
      <td>Player</td>
      <td>14</td>
      <td>CREATE TABLE `Player` (\n\t`id`\tINTEGER PRIMA...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>table</td>
      <td>Match</td>
      <td>Match</td>
      <td>18</td>
      <td>CREATE TABLE `Match` (\n\t`id`\tINTEGER PRIMAR...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>table</td>
      <td>League</td>
      <td>League</td>
      <td>24</td>
      <td>CREATE TABLE `League` (\n\t`id`\tINTEGER PRIMA...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>table</td>
      <td>Country</td>
      <td>Country</td>
      <td>26</td>
      <td>CREATE TABLE `Country` (\n\t`id`\tINTEGER PRIM...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>table</td>
      <td>Team</td>
      <td>Team</td>
      <td>29</td>
      <td>CREATE TABLE "Team" (\n\t`id`\tINTEGER PRIMARY...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>table</td>
      <td>Team_Attributes</td>
      <td>Team_Attributes</td>
      <td>2</td>
      <td>CREATE TABLE `Team_Attributes` (\n\t`id`\tINTE...</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

	countries = pd.read_sql("SELECT * FROM Country;", conn)
	countries.head()

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[47]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Belgium</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1729</td>
      <td>England</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4769</td>
      <td>France</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7809</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10257</td>
      <td>Italy</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

We reviewed the number of soccer teams in the world:

	teams= pd.read_sql("SELECT * FROM Team;", conn)
	teams.shape

(299, 5)

We noted that each team has a three letter country code.  The three letter code is something that we can merge upon with the European soccer teams .csv that we created!

	teams

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to unscroll output; double click to hide"></div><div class="output output_scroll"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[50]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_api_id</th>
      <th>team_fifa_api_id</th>
      <th>team_long_name</th>
      <th>team_short_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9987</td>
      <td>673.0</td>
      <td>KRC Genk</td>
      <td>GEN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>9993</td>
      <td>675.0</td>
      <td>Beerschot AC</td>
      <td>BAC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>10000</td>
      <td>15005.0</td>
      <td>SV Zulte-Waregem</td>
      <td>ZUL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>9994</td>
      <td>2007.0</td>
      <td>Sporting Lokeren</td>
      <td>LOK</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>9984</td>
      <td>1750.0</td>
      <td>KSV Cercle Brugge</td>
      <td>CEB</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>8635</td>
      <td>229.0</td>
      <td>RSC Anderlecht</td>
      <td>AND</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>9991</td>
      <td>674.0</td>
      <td>KAA Gent</td>
      <td>GEN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>9998</td>
      <td>1747.0</td>
      <td>RAEC Mons</td>
      <td>MON</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>7947</td>
      <td>NaN</td>
      <td>FCV Dender EH</td>
      <td>DEN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>9985</td>
      <td>232.0</td>
      <td>Standard de Liège</td>
      <td>STL</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>8203</td>
      <td>110724.0</td>
      <td>KV Mechelen</td>
      <td>MEC</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>8342</td>
      <td>231.0</td>
      <td>Club Brugge KV</td>
      <td>CLB</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>9999</td>
      <td>546.0</td>
      <td>KSV Roeselare</td>
      <td>ROS</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>8571</td>
      <td>100081.0</td>
      <td>KV Kortrijk</td>
      <td>KOR</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>4049</td>
      <td>NaN</td>
      <td>Tubize</td>
      <td>TUB</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>9996</td>
      <td>111560.0</td>
      <td>Royal Excel Mouscron</td>
      <td>MOU</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>10001</td>
      <td>681.0</td>
      <td>KVC Westerlo</td>
      <td>WES</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>9986</td>
      <td>670.0</td>
      <td>Sporting Charleroi</td>
      <td>CHA</td>
    </tr>
    <tr>
      <th>18</th>
      <td>614</td>
      <td>9997</td>
      <td>680.0</td>
      <td>Sint-Truidense VV</td>
      <td>STT</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1034</td>
      <td>9989</td>
      <td>239.0</td>
      <td>Lierse SK</td>
      <td>LIE</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1042</td>
      <td>6351</td>
      <td>2013.0</td>
      <td>KAS Eupen</td>
      <td>EUP</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1513</td>
      <td>1773</td>
      <td>100087.0</td>
      <td>Oud-Heverlee Leuven</td>
      <td>O-H</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2004</td>
      <td>8475</td>
      <td>110913.0</td>
      <td>Waasland-Beveren</td>
      <td>WAA</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2476</td>
      <td>8573</td>
      <td>682.0</td>
      <td>KV Oostende</td>
      <td>OOS</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2510</td>
      <td>274581</td>
      <td>111560.0</td>
      <td>Royal Excel Mouscron</td>
      <td>MOP</td>
    </tr>
    <tr>
      <th>25</th>
      <td>3457</td>
      <td>10260</td>
      <td>11.0</td>
      <td>Manchester United</td>
      <td>MUN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3458</td>
      <td>10261</td>
      <td>13.0</td>
      <td>Newcastle United</td>
      <td>NEW</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3459</td>
      <td>9825</td>
      <td>1.0</td>
      <td>Arsenal</td>
      <td>ARS</td>
    </tr>
    <tr>
      <th>28</th>
      <td>3460</td>
      <td>8659</td>
      <td>109.0</td>
      <td>West Bromwich Albion</td>
      <td>WBA</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3461</td>
      <td>8472</td>
      <td>106.0</td>
      <td>Sunderland</td>
      <td>SUN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>269</th>
      <td>43053</td>
      <td>9906</td>
      <td>240.0</td>
      <td>Atlético Madrid</td>
      <td>AMA</td>
    </tr>
    <tr>
      <th>270</th>
      <td>43054</td>
      <td>9864</td>
      <td>573.0</td>
      <td>Málaga CF</td>
      <td>MAL</td>
    </tr>
    <tr>
      <th>271</th>
      <td>43800</td>
      <td>9868</td>
      <td>1742.0</td>
      <td>Xerez Club Deportivo</td>
      <td>XER</td>
    </tr>
    <tr>
      <th>272</th>
      <td>43803</td>
      <td>8394</td>
      <td>244.0</td>
      <td>Real Zaragoza</td>
      <td>ZAR</td>
    </tr>
    <tr>
      <th>273</th>
      <td>43804</td>
      <td>9867</td>
      <td>260.0</td>
      <td>CD Tenerife</td>
      <td>TEN</td>
    </tr>
    <tr>
      <th>274</th>
      <td>44557</td>
      <td>10278</td>
      <td>100879.0</td>
      <td>Hércules Club de Fútbol</td>
      <td>HER</td>
    </tr>
    <tr>
      <th>275</th>
      <td>44565</td>
      <td>8581</td>
      <td>1853.0</td>
      <td>Levante UD</td>
      <td>LEV</td>
    </tr>
    <tr>
      <th>276</th>
      <td>44569</td>
      <td>8560</td>
      <td>457.0</td>
      <td>Real Sociedad</td>
      <td>SOC</td>
    </tr>
    <tr>
      <th>277</th>
      <td>45330</td>
      <td>7878</td>
      <td>110832.0</td>
      <td>Granada CF</td>
      <td>GRA</td>
    </tr>
    <tr>
      <th>278</th>
      <td>45333</td>
      <td>8370</td>
      <td>480.0</td>
      <td>Rayo Vallecano</td>
      <td>RAY</td>
    </tr>
    <tr>
      <th>279</th>
      <td>46087</td>
      <td>9910</td>
      <td>450.0</td>
      <td>RC Celta de Vigo</td>
      <td>CEL</td>
    </tr>
    <tr>
      <th>280</th>
      <td>46848</td>
      <td>10268</td>
      <td>468.0</td>
      <td>Elche CF</td>
      <td>ELC</td>
    </tr>
    <tr>
      <th>281</th>
      <td>47605</td>
      <td>8372</td>
      <td>467.0</td>
      <td>SD Eibar</td>
      <td>EIB</td>
    </tr>
    <tr>
      <th>282</th>
      <td>47612</td>
      <td>7869</td>
      <td>1867.0</td>
      <td>Córdoba CF</td>
      <td>COR</td>
    </tr>
    <tr>
      <th>283</th>
      <td>48358</td>
      <td>8306</td>
      <td>472.0</td>
      <td>UD Las Palmas</td>
      <td>LAS</td>
    </tr>
    <tr>
      <th>284</th>
      <td>49115</td>
      <td>9956</td>
      <td>322.0</td>
      <td>Grasshopper Club Zürich</td>
      <td>GRA</td>
    </tr>
    <tr>
      <th>285</th>
      <td>49116</td>
      <td>6493</td>
      <td>1714.0</td>
      <td>AC Bellinzona</td>
      <td>BEL</td>
    </tr>
    <tr>
      <th>286</th>
      <td>49117</td>
      <td>10192</td>
      <td>900.0</td>
      <td>BSC Young Boys</td>
      <td>YB</td>
    </tr>
    <tr>
      <th>287</th>
      <td>49118</td>
      <td>9931</td>
      <td>896.0</td>
      <td>FC Basel</td>
      <td>BAS</td>
    </tr>
    <tr>
      <th>288</th>
      <td>49119</td>
      <td>9930</td>
      <td>434.0</td>
      <td>FC Aarau</td>
      <td>AAR</td>
    </tr>
    <tr>
      <th>289</th>
      <td>49120</td>
      <td>10179</td>
      <td>110770.0</td>
      <td>FC Sion</td>
      <td>SIO</td>
    </tr>
    <tr>
      <th>290</th>
      <td>49121</td>
      <td>10199</td>
      <td>897.0</td>
      <td>FC Luzern</td>
      <td>LUZ</td>
    </tr>
    <tr>
      <th>291</th>
      <td>49122</td>
      <td>9824</td>
      <td>286.0</td>
      <td>FC Vaduz</td>
      <td>VAD</td>
    </tr>
    <tr>
      <th>292</th>
      <td>49123</td>
      <td>7955</td>
      <td>435.0</td>
      <td>Neuchâtel Xamax</td>
      <td>XAM</td>
    </tr>
    <tr>
      <th>293</th>
      <td>49124</td>
      <td>10243</td>
      <td>894.0</td>
      <td>FC Zürich</td>
      <td>ZUR</td>
    </tr>
    <tr>
      <th>294</th>
      <td>49479</td>
      <td>10190</td>
      <td>898.0</td>
      <td>FC St. Gallen</td>
      <td>GAL</td>
    </tr>
    <tr>
      <th>295</th>
      <td>49837</td>
      <td>10191</td>
      <td>1715.0</td>
      <td>FC Thun</td>
      <td>THU</td>
    </tr>
    <tr>
      <th>296</th>
      <td>50201</td>
      <td>9777</td>
      <td>324.0</td>
      <td>Servette FC</td>
      <td>SER</td>
    </tr>
    <tr>
      <th>297</th>
      <td>50204</td>
      <td>7730</td>
      <td>1862.0</td>
      <td>FC Lausanne-Sports</td>
      <td>LAU</td>
    </tr>
    <tr>
      <th>298</th>
      <td>51606</td>
      <td>7896</td>
      <td>NaN</td>
      <td>Lugano</td>
      <td>LUG</td>
    </tr>
  </tbody>
</table>
<p>299 rows × 5 columns</p>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

	players = pd.read_sql("SELECT * FROM Player;", conn)
	players.head()

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>player_api_id</th>
      <th>player_name</th>
      <th>player_fifa_api_id</th>
      <th>birthday</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>505942</td>
      <td>Aaron Appindangoye</td>
      <td>218353</td>
      <td>1992-02-29 00:00:00</td>
      <td>182.88</td>
      <td>187</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>155782</td>
      <td>Aaron Cresswell</td>
      <td>189615</td>
      <td>1989-12-15 00:00:00</td>
      <td>170.18</td>
      <td>146</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>162549</td>
      <td>Aaron Doran</td>
      <td>186170</td>
      <td>1991-05-13 00:00:00</td>
      <td>170.18</td>
      <td>163</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>30572</td>
      <td>Aaron Galindo</td>
      <td>140161</td>
      <td>1982-05-08 00:00:00</td>
      <td>182.88</td>
      <td>198</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>23780</td>
      <td>Aaron Hughes</td>
      <td>17725</td>
      <td>1979-11-08 00:00:00</td>
      <td>182.88</td>
      <td>154</td>
    </tr>
  </tbody>
</table>

	player_att = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
	player_att.head()

<div class="cell code_cell rendered selected" tabindex="2"><div class="input"><div class="prompt input_prompt"><bdi>In</bdi>&nbsp;[52]:</div><div class="inner_cell"><div class="ctb_hideshow"><div class="celltoolbar"></div></div><div class="input_area"><div class="CodeMirror cm-s-ipython"><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 1px; left: 6px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-vscrollbar" cm-not-content="true"><div style="min-width: 1px; height: 0px;"></div></div><div class="CodeMirror-hscrollbar" cm-not-content="true"><div style="height: 100%; min-height: 1px; width: 0px;"></div></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; min-width: 116px; margin-bottom: -17px; border-right-width: 13px; min-height: 28px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"></div><div class="CodeMirror-measure"></div><div style="position: relative; z-index: 1;"></div><div class="CodeMirror-cursors"><div class="CodeMirror-cursor" style="left: 6px; top: 0px; height: 17px;">&nbsp;</div></div><div class="CodeMirror-code" role="presentation"><pre class=" CodeMirror-line " role="presentation"><span role="presentation" style="padding-right: 0.1px;"><span class="cm-variable">players</span>.<span class="cm-property">head</span>()</span></pre></div></div></div></div></div><div style="position: absolute; height: 13px; width: 1px; border-bottom: 0px solid transparent; top: 28px;"></div><div class="CodeMirror-gutters" style="display: none; height: 41px;"></div></div></div></div></div></div><div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[52]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>player_api_id</th>
      <th>player_name</th>
      <th>player_fifa_api_id</th>
      <th>birthday</th>
      <th>height</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>505942</td>
      <td>Aaron Appindangoye</td>
      <td>218353</td>
      <td>1992-02-29 00:00:00</td>
      <td>182.88</td>
      <td>187</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>155782</td>
      <td>Aaron Cresswell</td>
      <td>189615</td>
      <td>1989-12-15 00:00:00</td>
      <td>170.18</td>
      <td>146</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>162549</td>
      <td>Aaron Doran</td>
      <td>186170</td>
      <td>1991-05-13 00:00:00</td>
      <td>170.18</td>
      <td>163</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>30572</td>
      <td>Aaron Galindo</td>
      <td>140161</td>
      <td>1982-05-08 00:00:00</td>
      <td>182.88</td>
      <td>198</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>23780</td>
      <td>Aaron Hughes</td>
      <td>17725</td>
      <td>1979-11-08 00:00:00</td>
      <td>182.88</td>
      <td>154</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div></div>

	data = pd.read_csv("Europe/teams.csv")
	data.head(10)

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[56]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column1</th>
      <th>Column2</th>
      <th>Column3</th>
      <th>Column4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aut</td>
      <td>Austria</td>
      <td>AUT</td>
      <td>at</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bel</td>
      <td>Belgium</td>
      <td>BEL</td>
      <td>be</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cyp</td>
      <td>Cyprus</td>
      <td>CYP</td>
      <td>cy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ger</td>
      <td>Germany</td>
      <td>GER</td>
      <td>de</td>
    </tr>
    <tr>
      <th>4</th>
      <td>est</td>
      <td>Estonia</td>
      <td>EST</td>
      <td>ee</td>
    </tr>
    <tr>
      <th>5</th>
      <td>esp</td>
      <td>Spain</td>
      <td>ESP</td>
      <td>es</td>
    </tr>
    <tr>
      <th>6</th>
      <td>fin</td>
      <td>Finland</td>
      <td>FIN</td>
      <td>fi</td>
    </tr>
    <tr>
      <th>7</th>
      <td>fra</td>
      <td>France</td>
      <td>FRA</td>
      <td>fr</td>
    </tr>
    <tr>
      <th>8</th>
      <td>gre</td>
      <td>Greece</td>
      <td>GRE</td>
      <td>gr</td>
    </tr>
    <tr>
      <th>9</th>
      <td>irl</td>
      <td>Ireland</td>
      <td>IRL</td>
      <td>ie</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

# Merging Database

	raw_data = {
        'id': ['1', '2', '3', '4', '5'],
        'team_long_name': ['Austria', 'Belgium', 'Cyprus', 'Germany', 'Estonia'], 
      'team_short_name': ['AUT', 'BEL', 'CYP', 'GER', 'EST']}
	df_a = pd.DataFrame(raw_data, columns = ['id', 'team_long_name', 'team_short_name'])
	df_a

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[57]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_long_name</th>
      <th>team_short_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Austria</td>
      <td>AUT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Belgium</td>
      <td>BEL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Cyprus</td>
      <td>CYP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Germany</td>
      <td>GER</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Estonia</td>
      <td>EST</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

	raw_data = {
        'id': ['4', '5', '6', '7', '8'],
        'team_long_name': ['KSV Cercle Brugge', 'RSC Anderlecht', 'KAA Gent', 'RAEC Mons', 'FCV Dender EH'], 
        'team_short_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']}
	df_b = pd.DataFrame(raw_data, columns = ['id', 'team_long_name', 'team_short_name'])
	df_b

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[58]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_long_name</th>
      <th>team_short_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>KSV Cercle Brugge</td>
      <td>Bonder</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>RSC Anderlecht</td>
      <td>Black</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>KAA Gent</td>
      <td>Balwner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>RAEC Mons</td>
      <td>Brice</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>FCV Dender EH</td>
      <td>Btisan</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

	raw_data = {
        'id': ['1', '2', '3', '4', '5', '7', '8', '9',],
        'team_fifa_api_id': [673.0, 675.0, 15005.0, 2007.0, 1750.0, 229.0, 674.0, 1747.0]}
	df_n = pd.DataFrame(raw_data, columns = ['id','team_fifa_api_id'])
	df_n

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[59]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_fifa_api_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>673.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>675.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15005.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1750.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>229.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>674.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>1747.0</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

	df_new = pd.concat([df_a, df_b])
	df_new

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[61]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_long_name</th>
      <th>team_short_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Austria</td>
      <td>AUT</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Belgium</td>
      <td>BEL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Cyprus</td>
      <td>CYP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Germany</td>
      <td>GER</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Estonia</td>
      <td>EST</td>
    </tr>
    <tr>
      <th>0</th>
      <td>4</td>
      <td>KSV Cercle Brugge</td>
      <td>Bonder</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>RSC Anderlecht</td>
      <td>Black</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>KAA Gent</td>
      <td>Balwner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>RAEC Mons</td>
      <td>Brice</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>FCV Dender EH</td>
      <td>Btisan</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

	pd.concat([df_a, df_b], axis=1)

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[62]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_long_name</th>
      <th>team_short_name</th>
      <th>id</th>
      <th>team_long_name</th>
      <th>team_short_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Austria</td>
      <td>AUT</td>
      <td>4</td>
      <td>KSV Cercle Brugge</td>
      <td>Bonder</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Belgium</td>
      <td>BEL</td>
      <td>5</td>
      <td>RSC Anderlecht</td>
      <td>Black</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Cyprus</td>
      <td>CYP</td>
      <td>6</td>
      <td>KAA Gent</td>
      <td>Balwner</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Germany</td>
      <td>GER</td>
      <td>7</td>
      <td>RAEC Mons</td>
      <td>Brice</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Estonia</td>
      <td>EST</td>
      <td>8</td>
      <td>FCV Dender EH</td>
      <td>Btisan</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

	pd.merge(df_new, df_n, on='id')

<div class="output_wrapper"><div class="out_prompt_overlay prompt" title="click to scroll output; double click to hide"></div><div class="output"><div class="output_area"><div class="prompt output_prompt"><bdi>Out[63]:</bdi></div><div class="output_subarea output_html rendered_html output_result"><div>
<style scoped="">
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_long_name</th>
      <th>team_short_name</th>
      <th>team_fifa_api_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Austria</td>
      <td>AUT</td>
      <td>673.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Belgium</td>
      <td>BEL</td>
      <td>675.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Cyprus</td>
      <td>CYP</td>
      <td>15005.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Germany</td>
      <td>GER</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>KSV Cercle Brugge</td>
      <td>Bonder</td>
      <td>2007.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>Estonia</td>
      <td>EST</td>
      <td>1750.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>RSC Anderlecht</td>
      <td>Black</td>
      <td>1750.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>RAEC Mons</td>
      <td>Brice</td>
      <td>229.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>FCV Dender EH</td>
      <td>Btisan</td>
      <td>674.0</td>
    </tr>
  </tbody>
</table>
</div></div></div></div><div class="btn btn-default output_collapsed" title="click to expand output" style="display: none;">. . .</div></div>

# Create Engine

	connection_string = "database1.sqlite"
	e = create_engine('sqlite://') 

	e.table_names()

[]