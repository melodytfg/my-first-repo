# my-first-repo
week 4
import streamlit as st 
import streamlit.components.v1 as stc 
from eda_app import run_eda_app
from ml_app import run_ml_app

html_temp = """
		<div style="background-color:#3872fb;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">455皓軒Demo1 Early Stage DM Risk Data App </h1>
		<h4 style="color:white;text-align:center;">Diabetes </h4>
		</div>
		"""

def main():
	# st.title("ML Web App with Streamlit")
	stc.html(html_temp)

	menu = ["Home","EDA","ML","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		st.subheader("Home")
		st.write("""
			### Early Stage Diabetes Risk Predictor App
			This dataset contains the sign and symptoms data of newly diabetic or would be diabetic patient.
			#### Datasource
				- https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset.
			#### App Content
				- EDA Section: Exploratory Data Analysis of Data
				- ML Section: ML Predictor App

			""")
	elif choice == "EDA":
		run_eda_app()
	elif choice == "ML":
		run_ml_app()
	else:
		st.subheader("About")
		st.text("Learn Streamlit Course")
		st.text("Jesus Saves @JCharisTech")
		st.text("By Jesse E.Agbe(JCharis)")

if __name__ == '__main__':
	main()
	
import streamlit as st 
import pandas as pd 

# Data Viz Pkgs

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import plotly.express as px 


@st.cache
def load_data(data):
	df = pd.read_csv(data)
	return df


def run_eda_app():
	st.subheader("EDA Section")
	df = load_data("data/diabetes_data_upload.csv")
	df_clean = load_data("data/diabetes_data_upload_clean.csv")
	freq_df = load_data("data/freqdist_of_age_data.csv")

	submenu = st.sidebar.selectbox("SubMenu",["Descriptive","Plots"])
	if submenu == "Descriptive":
		
		st.dataframe(df)

		with st.expander("Data Types Summary"):
			st.dataframe(df.dtypes)

		with st.expander("Descriptive Summary"):
			st.dataframe(df_clean.describe())

		with st.expander("Gender Distribution"):
			st.dataframe(df['Gender'].value_counts())

		with st.expander("Class Distribution"):
			st.dataframe(df['class'].value_counts())
	else:
		st.subheader("Plots")

		# Layouts
		col1,col2 = st.columns([2,1])
		with col1:
			with st.expander("Dist Plot of Gender"):
				# fig = plt.figure()
				# sns.countplot(df['Gender'])
				# st.pyplot(fig)

				gen_df = df['Gender'].value_counts().to_frame()
				gen_df = gen_df.reset_index()
				gen_df.columns = ['Gender Type','Counts']
				# st.dataframe(gen_df)
				p01 = px.pie(gen_df,names='Gender Type',values='Counts')
				st.plotly_chart(p01,use_container_width=True)

			with st.expander("Dist Plot of Class"):
				fig = plt.figure()
				sns.countplot(df['class'])
				st.pyplot(fig)





		with col2:
			with st.expander("Gender Distribution"):
				st.dataframe(df['Gender'].value_counts())

			with st.expander("Class Distribution"):
				st.dataframe(df['class'].value_counts())
			
		with st.expander("Frequency Dist Plot of Age"):
			# fig,ax = plt.subplots()
			# ax.bar(freq_df['Age'],freq_df['count'])
			# plt.ylabel('Counts')
			# plt.title('Frequency Count of Age')
			# plt.xticks(rotation=45)
			# st.pyplot(fig)

			p = px.bar(freq_df,x='Age',y='count')
			st.plotly_chart(p)

			p2 = px.line(freq_df,x='Age',y='count')
			st.plotly_chart(p2)

		with st.expander("Outlier Detection Plot"):
			# outlier_df = 
			fig = plt.figure()
			sns.boxplot(df['Age'])
			st.pyplot(fig)

			p3 = px.box(df,x='Age',color='Gender')
			st.plotly_chart(p3)

		with st.expander("Correlation Plot"):
			corr_matrix = df_clean.corr()
			fig = plt.figure(figsize=(20,10))
			sns.heatmap(corr_matrix,annot=True)
			st.pyplot(fig)

			p3 = px.imshow(corr_matrix)
			st.plotly_chart(p3)


import streamlit as st 
import joblib
import os
import numpy as np

attrib_info = """
#### Attribute Information:
    - Age 1.20-65
    - Sex 1. Male, 2.Female
    - Polyuria 1.Yes, 2.No.
    - Polydipsia 1.Yes, 2.No.
    - sudden weight loss 1.Yes, 2.No.
    - weakness 1.Yes, 2.No.
    - Polyphagia 1.Yes, 2.No.
    - Genital thrush 1.Yes, 2.No.
    - visual blurring 1.Yes, 2.No.
    - Itching 1.Yes, 2.No.
    - Irritability 1.Yes, 2.No.
    - delayed healing 1.Yes, 2.No.
    - partial paresis 1.Yes, 2.No.
    - muscle sti
ness 1.Yes, 2.No.
    - Alopecia 1.Yes, 2.No.
    - Obesity 1.Yes, 2.No.
    - Class 1.Positive, 2.Negative.

"""
label_dict = {"No":0,"Yes":1}
gender_map = {"Female":0,"Male":1}
target_label_map = {"Negative":0,"Positive":1}

['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
       'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
       'itching', 'irritability', 'delayed_healing', 'partial_paresis',
       'muscle_stiffness', 'alopecia', 'obesity', 'class']


def get_fvalue(val):
	feature_dict = {"No":0,"Yes":1}
	for key,value in feature_dict.items():
		if val == key:
			return value 

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 



# Load ML Models
@st.cache_data
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


def run_ml_app():
	st.subheader("Machine Learning Section")
	loaded_model = load_model("models/logistic_regression_model_diabetes_21_oct_2020.pkl")

	with st.expander("Attributes Info"):
		st.markdown(attrib_info,unsafe_allow_html=True)

	# Layout
	col1,col2 = st.columns(2)

	with col1:
		age = st.number_input("Age",10,100)
		gender = st.radio("Gender",("Female","Male"))
		polyuria = st.radio("Polyuria",["No","Yes"])
		polydipsia = st.radio("Polydipsia",["No","Yes"]) 
		sudden_weight_loss = st.selectbox("Sudden_weight_loss",["No","Yes"])
		weakness = st.selectbox("weakness",["No","Yes"]) 
		polyphagia = st.selectbox("polyphagia",["No","Yes"]) 
		genital_thrush = st.selectbox("Genital_thrush",["No","Yes"]) 
		# genital_thrush = st.select_slider("Genital_thrush",["No","Yes"]) 
		
	
	with col2:
		visual_blurring = st.selectbox("Visual_blurring",["No","Yes"])
		itching = st.radio("itching",["No","Yes"]) 
		irritability = st.radio("irritability",["No","Yes"]) 
		delayed_healing = st.radio("delayed_healing",["No","Yes"]) 
		partial_paresis = st.selectbox("Partial_paresis",["No","Yes"])
		muscle_stiffness = st.select_slider("muscle_stiffness",["No","Yes"]) 
		alopecia = st.select_slider("alopecia",["No","Yes"]) 
		obesity = st.select_slider("obesity",["No","Yes"]) 
		# obesity = st.selectbox("obesity",["No","Yes"]) 

	with st.expander("Your Selected Options"):
		result = {'age':age,
		'gender':gender,
		'polyuria':polyuria,
		'polydipsia':polydipsia,
		'sudden_weight_loss':sudden_weight_loss,
		'weakness':weakness,
		'polyphagia':polyphagia,
		'genital_thrush':genital_thrush,
		'visual_blurring':visual_blurring,
		'itching':itching,
		'irritability':irritability,
		'delayed_healing':delayed_healing,
		'partial_paresis':partial_paresis,
		'muscle_stiffness':muscle_stiffness,
		'alopecia':alopecia,
		'obesity':obesity}
		st.write(result)
		encoded_result = []
		for i in result.values():
			if type(i) == int:
				encoded_result.append(i)
			elif i in ["Female","Male"]:
				res = get_value(i,gender_map)
				encoded_result.append(res)
			else:
				encoded_result.append(get_fvalue(i))


		# st.write(encoded_result)
	with st.expander("Prediction Results"):
		single_sample = np.array(encoded_result).reshape(1,-1)

		
		prediction = loaded_model.predict(single_sample)
		pred_prob = loaded_model.predict_proba(single_sample)
		st.write(prediction)
		if prediction == 1:
			st.warning("Positive Risk-{}".format(prediction[0]))
			pred_probability_score = {"Negative DM":pred_prob[0][0]*100,"Positive DM":pred_prob[0][1]*100}
			st.subheader("Prediction Probability Score")
			st.json(pred_probability_score)
		else:
			st.success("Negative Risk-{}".format(prediction[0]))
			pred_probability_score = {"Negative DM":pred_prob[0][0]*100,"Positive DM":pred_prob[0][1]*100}
			st.subheader("Prediction Probability Score")
			st.json(pred_probability_score)
