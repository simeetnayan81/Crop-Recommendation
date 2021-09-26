import streamlit as st
import sklearn
import joblib
import pandas as pd

def pred(feat):
	feat=pd.DataFrame(feat)
	clf=joblib.load('GNBClassifier.pkl')
	pred=clf.predict(feat)
	s= "Recommended crop is " + str(pred[0])
	st.write(s)


def main():
	st.title("Crop Recommendation App")
	N=st.text_input("Ratio of Nitrogen Content in Soil", "")
	P=st.text_input("Ratio of Phosphorous Content in Soil", "")
	K=st.text_input("Ratio of  Potassium Content in Soil", "")
	temp=st.text_input("Temperature", "")
	hum=st.text_input("Humidity", "")
	ph=st.text_input("pH", "")
	rain=st.text_input("Rainfall", "")
	x=st.button("Reccomend")
	l=[[N, P, K, temp, hum, ph, rain]]
	if x:
		pred(l)

if __name__=='__main__':
	main()