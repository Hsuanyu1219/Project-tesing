import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# Pandas library for the pandas dataframes
import pandas as pd    
import numpy as np

# Import Scikit-Learn library for decision tree models
import sklearn         
from sklearn import linear_model, datasets
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


# Import plotting libraries
import seaborn as sns
import matplotlib 
from matplotlib import pyplot as plt

# Set larger fontsize for all plots
matplotlib.rcParams.update({'font.size': 18})
from IPython.display import clear_output
 





header = st.container()
dataset =st.container()
features = st.container()
model_training = st.container()





# Cashe or Cashing: For large dataset, avoid rerunning out of huge memory, let only run once!

@st.cache
def get_data(filename): 
	Cheetah_data = pd.read_csv(filename)

	return Cheetah_data

with header:
	st.title('Welcome to our project!') 
	st.text('IN this project, we propose idea into trainig image and fixing ')

with dataset:
	st.header('NYC taxi dataset')
	st.text('The dataset is all from NTHU@MSE')
	Cheetah_data = get_data('data/running_cheetah.csv')
	st.write(Cheetah_data.head())


	st.subheader('Pick up fraction of pce from HDEdataset')
	pce_dist = pd.DataFrame(Cheetah_data['time [hr]'].value_counts()).head(10)
	st.bar_chart(pce_dist)


with features:
	st.header('Feature I create')

	st.markdown('* **first featues:** I create this feature for .....')
	st.markdown('* **second featues:** I create this feature for .....')




with model_training:
	st.header('Time to train model')
	st.text('here you choose the model and reason for why doing this !')

	sel_col, disp_col = st.columns(2)

	max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value = 20, step =10)

	n_estimators = sel_col.selectbox('How many trees shold there be?', options=[100,200,300,'No limit'], index=0)

	sel_col.text('Here is the list of feature in my data')
	sel_col.table(Cheetah_data.columns)

	input_feature = sel_col.text_input('Which feature should be used as the input feature?','time [hr]') 

	#input_feature = sel_col.selectbox('Which feature should be used as the input feature?',options=['pce','mass','voc','jsc'], index=0)


	if n_estimators == 'No limit':
		regr = RandomForestRegressor(max_depth=max_depth)
	else:
		regr = RandomForestRegressor(max_depth=max_depth, n_estimators= n_estimators)

	X=Cheetah_data[[input_feature]].values
	y=Cheetah_data[['speed [miles/hr]']].values

	regr.fit(X,y)
	prediction = regr.predict(y)

	disp_col.subheader('Mean absolute error of the model is:')
	disp_col.write(mean_absolute_error(y,prediction))

	disp_col.subheader('Mean squared error of the model is:')
	disp_col.write(mean_squared_error(y,prediction))


	disp_col.subheader('R squared error of the model is:')
	disp_col.write(r2_score(y,prediction))









