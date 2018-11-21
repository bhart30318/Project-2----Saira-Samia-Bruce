# Preface
We will be using an open dataset from the popular site Kaggle. This European Soccer Database has more than 25,000 matches and more than 10,000 players for European professional soccer seasons from 2008 to 2016. 

# Dataset
We will be using an open dataset from the popular site Kaggle. This European Soccer Database has more than 25,000 matches and more than 10,000 players for European professional soccer seasons from 2008 to 2016.

Although we won’t be getting into the details of it for our example, the dataset even has attributes on weekly game updates, team line up, and detailed match events.

The goal of this notebook is to walk you through an end to end process of analyzing a dataset and introduce you to what we will be covering in this course. Our simple analytical process will include some steps for exploring and cleaning our dataset, some steps for predicting player performance using basic statistics, and some steps for grouping similar clusters using machine learning.

# Getting Started

To get started, we will need to:

1. Download the data from: https://www.kaggle.com/hugomathien/soccer
2. Extract the zip file called "soccer.zip"

# Import Libraries
We will start by importing the Python libraries we will be using in this analysis. These libraries include: sqllite3 for interacting with a local relational database pandas and numpy for data ingestion and manipulation matplotlib for data visualization specific methods from sklearn for Machine Learning and customplot, which contains custom functions we have written for this notebook

	import sqlite3
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.cluster import KMeans
	from sklearn.preprocessing import scale
	from customplot import *

# Ingest Data
Now, we will need to read the dataset using the commands below.

Note: Make sure you run the import cell above (shift+enter) before you run the data ingest code below.

df is a variable pointing to a pandas data frame.

Create your connection.
	cnx = sqlite3.connect('database.sqlite')
	df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)

# Exploring Data
We will start our data exploration by generating simple statistics of the data. Let us look at what the data columns are using a pandas attribute called “columns”.

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


Next will display simple statistics of our dataset. You need to run each cell to make sure you see the outputs.

# Data Cleaning: Handling Missing Data
Real data is never clean. We need to make sure we clean the data by converting or getting rid of null or missing values.
The next code cell will show you if any of the 183978 rows have null value in one of the 42 columns.

is any row NULL ?

	df.isnull().any().any(), df.shape
	(True, (183978, 42))

Now let’s try to find how many data points in each column are null.

Fixing Null Values by Deleting Them
In our next two lines, we will drop the null values by going through each row.

Fix it

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

To find exactly how many lines we removed, we need to subtract the current number of rows in our data frame from the original number of rows.

How many rows with NULL values?

	rows - df.shape[0]
	3624

Our data table has many lines as you have seen. We can only look at few lines at once. Instead of looking at same top 10 lines every time, we shuffle - so we get to see different random sample on top. This way, we make sure the data is not in any particular order when we try sampling from it (like taking top or bottom few rows) by randomly shuffling the rows.

Shuffle the rows of df so we get a distributed sample when we display top few rows

	df = df.reindex(np.random.permutation(df.index))

Predicting: ‘overall_rating’ of a player, now that our data cleaning step is reasonably complete and we can trust and understand the data more, we will start diving into the dataset further.

Let’s take a look at top few rows.

We will use the head function for data frames for this task. This gives us every column in every row.

	df.head(5)

	5 rows × 42 columns

Most of the time, we are only interested in plotting some columns. In that case, we can use the pandas column selection option as follows. Please ignore the first column in the output of the one line code below. It is the unique identifier that acts as an index for the data.

Note: From this point on, we will start referring to the columns as “features” in our description.

	df[:10][['penalties', 'overall_rating']]

# Feature Correlation Analysis
Next, we will check if ‘penalties’ is correlated to ‘overall_rating’. We are using a similar selection operation, bu this time for all the rows and within the correlation function.

Are these correlated (using Pearson’s correlation coefficient) ?

	df['overall_rating'].corr(df['penalties'])

We see that Pearson’s Correlation Coefficient for these two columns is 0.39. 

Pearson goes from -1 to +1. A value of 0 would have told there is no correlation, so we shouldn’t bother looking at that attribute. A value of 0.39 shows some correlation, although it could be stronger. 

At least, we have these attributes which are slightly correlated. This gives us hope that we might be able to build a meaningful predictor using these ‘weakly’ correlated features.

Next, we will create a list of features that we would like to iterate the same operation on.

Create a list of potential Features that you want to measure correlation with

	potentialFeatures = ['acceleration', 'curve', 'free_kick_accuracy', 'ball_control', 'shot_power', 'stamina']

The for loop below prints out the correlation coefficient of “overall_rating” of a player with each feature we added to the list as potential.

check how the features are correlated with the overall ratings

	for f in potentialFeatures:
    	related = df['overall_rating'].corr(df[f])
    	print("%s: %f" % (f,related))

	acceleration: 0.243998
	curve: 0.357566
	free_kick_accuracy: 0.349800
	ball_control: 0.443991
	shot_power: 0.428053
	stamina: 0.325606

Which features have the highest correlation with overall_rating?

Looking at the values printed by the previous cell, we notice that the to two are “ball_control” (0.44) and “shot_power” (0.43). So these two features seem to have higher correlation with “overall_rating”.

Data Visualization:
Next we will start plotting the correlation coefficients of each feature with “overall_rating”. We start by selecting the columns and creating a list with correlation coefficients, called “correlations”.

	cols = ['potential',  'crossing', 'finishing', 'heading_accuracy',
	       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
	       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
	       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
	       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
	       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
	       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
	       'gk_reflexes']

create a list containing Pearson's correlation between 'overall_rating' with each column in cols

	correlations = [ df['overall_rating'].corr(df[f]) for f in cols ]
	len(cols), len(correlations)
	(34, 34)
	
We make sure that the number of selected features and the correlations calculated are the same, e.g., both 34 in this case. Next couple of cells show some lines of code that use pandas plaotting functions to create a 2D graph of these correlation vealues and column names.

create a function for plotting a dataframe with string columns and numeric values

	def plot_dataframe(df, y_label):  
    color='coral'
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    plt.ylabel(y_label)

    ax = df2.correlation.plot(linewidth=3.3, color=color)
    ax.set_xticks(df2.index)
    ax.set_xticklabels(df2.attributes, rotation=75); #Notice the ; (remove it and see what happens !)
    plt.show()
create a dataframe using cols and correlations

	df2 = pd.DataFrame({'attributes': cols, 'correlation': correlations}) 
let's plot above dataframe using the function we created
    
	plot_dataframe(df2, 'Player\'s Overall Rating')





```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/bhart30318/Project-2----Saira-Samia-Bruce/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
