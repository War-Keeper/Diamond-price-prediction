import xgboost as xgb
import streamlit as st
import pandas as pd
import sklearn
from pickle

#Loading up the Regression model we created
xgbr_D = xgb.XGBRegressor()
xgbr_S = xgb.XGBRegressor()
xgbr_D.load_model('xgbr_DiamondSE.json')
xgbr_S.load_model('xgbr_StoneAlgo.json')

linear_D = 0
linear_S = 0 
knnr_D = 0
knnr_S = 0

with open('linear_DiamondSE.pkl', 'rb') as f:
  linear_D = pickle.load(f)
with open('linear_StoneAlgo.pkl', 'rb') as f:
  linear_S = pickle.load(f)
with open('knnr_DiamondSE.pkl', 'rb') as f:
  knnr_D = pickle.load(f)
with open('knnr_StoneAlgo.pkl', 'rb') as f:
  knnr_S = pickle.load(f)

#Caching the model for faster loading
@st.cache

def predict_StoneAlgo(shape, carat, cut, color, clarity, polish, symmetry, fluorescence):
    #Predicting the price of the carat
    cut_dict = {'Excellent': 5,
        'Very Good': 4,
        'Good': 3,
        'Fair': 2,
        'Ideal': 5.5}
    
    cut = cut_dict[cut]

    shape_dict = {'Round': 11,
        'Cushion': 10,
        'Princess': 9,
        'Emerald': 8,
        'Oval': 7,
        'Pear': 6,
        'Marquise': 5,
        'Radiant': 4,
        'Heart': 3,
        'Asscher': 2,
        'CM': 1}
    
    shape = shape_dict[shape]

    color_dict = {'D': 15,
        'E': 14,
        'F': 13,
        'G': 12,
        'H': 11,
        'I': 10,
        'J': 9,
        'K': 8,
        'L': 7,
        'M': 6,
        'N': 5,
        'YZ': 1,
        'WX': 2,
        'OP': 4,
        'UV': 3,
        'Y': 1.5}
    
    color = color_dict[color]
    
    clarity_dict = {'VS2': 4,
        'VS1': 5,
        'SI1': 3,
        'VVS2': 6,
        'SI2': 2,
        'VVS1': 7,
        'IF': 8,
        'I1': 1,
        'FL': 9,
        'None': 0}
    
    clarity = clarity_dict[clarity]
    
    polish_dict = {'Excellent': 5,
        'Very Good': 4,
        'Good': 3,
        'Ideal': 5.5,
        'Fair': 2,
        'Very Good-Excel': 4.5,
        'Good-Very Good': 3.5}
    
    polish = polish_dict[polish]
    
    symmetry_dict = {'Excellent': 5,
        'Very Good': 4,
        'Good': 3,
        'Fair': 2,
        'Ideal': 5.5,
        'Poor': 1}
    
    symmetry = symmetry_dict[symmetry]
    
    fluorescence_dict = {'None': 5,
        'Faint': 4,
        'Medium': 3,
        'Strong': 2,
        'Very Slight': 4.5,
        'Very Strong': 1,
        'Slight': 4,
        'Medium Blue': 3,
        'Strong Blue': 2,
        'Very Strong Blue': 1,
        'Medium White': 3}
    
    fluorescence = fluorescence_dict[fluorescence]

    df = pd.DataFrame([[shape, carat, cut, color, clarity, polish, symmetry, fluorescence]], columns=['shape', 'carat', 'cut', 'color', 'clarity', 'polish', 'symmetry', 'fluorescence'])

    pred_linear = linear_S.predict(df)
    
    pred_knnr = knnr_S.predict(df)

    pred_xgbr = xgbr_S.predict(df)
    
    return pred_linear, pred_knnr, pred_xgbr

def predict_DiamondSE(shape, carat, cut, color, clarity, width, depth, x, y, z):
    #Predicting the price of the carat
    cut_dict = {'Excellent': 5,
        'Very Good': 4,
        'Good': 3,
        'Fair': 2,
        'Ideal': 5.5}
    
    cut = cut_dict[cut]

    shape_dict = {'Round': 11,
        'Cushion': 10,
        'Princess': 9,
        'Emerald': 8,
        'Oval': 7,
        'Pear': 6,
        'Marquise': 5,
        'Radiant': 4,
        'Heart': 3,
        'Asscher': 2,
        'CM': 1}
    
    shape = shape_dict[shape]

    color_dict = {'D': 15,
        'E': 14,
        'F': 13,
        'G': 12,
        'H': 11,
        'I': 10,
        'J': 9,
        'K': 8,
        'L': 7,
        'M': 6,
        'N': 5,
        'YZ': 1,
        'WX': 2,
        'OP': 4,
        'UV': 3,
        'Y': 1.5}
    
    color = color_dict[color]
    
    clarity_dict = {'VS2': 4,
        'VS1': 5,
        'SI1': 3,
        'VVS2': 6,
        'SI2': 2,
        'VVS1': 7,
        'IF': 8,
        'I1': 1,
        'FL': 9,
        'None': 0}
    
    clarity = clarity_dict[clarity]

    df = pd.DataFrame([[shape, carat, cut, color, clarity, width, depth, x, y, z]], columns=['shape', 'carat', 'cut', 'color', 'clarity', 'width', 'depth', 'x', 'y', 'z'])

    pred_linear = linear_D.predict(df)
    
    pred_knnr = knnr_D.predict(df)

    pred_xgbr = xgbr_D.predict(df)
    
    return pred_linear, pred_knnr, pred_xgbr

st.title('Diamond Price Predictor')
st.image("""https://images.alphacoders.com/689/689045.jpg""")
st.header('Enter the characteristics of the diamond:')
st.text('Either enter:   shape, carat, cut, color, clarity, polish, symmetry, fluorescence\n \
         or:             shape, carat, cut, color, clarity, width, depth, x, y, z \n or all of the info.')

st.text('if not sufficient amount of information is entered, then preditions will NOT be accurate.')

shape = st.selectbox('Shape:', ['Round', 'Cushion', 'Princess', 'Emerald', 'Oval', 'Pear', 'Marquise', 'Radiant', 'Heart', 'Asscher', 'CM'])
carat = st.number_input('Carat Weight:', min_value=0, max_value=100.0, value=0)
cut = st.selectbox('Cut Rating:', ['Fair', 'Good', 'Very Good', 'Excellent', 'Ideal'])
color = st.selectbox('Color Rating:', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
clarity = st.selectbox('Clarity Rating:', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
polish = st.selectbox('Polish Rating:', ['Ideal', 'Excellent', 'Very Good', 'Good', 'Fair', 'Good-Very Good', 'Very Good-Excel'])
symmetry = st.selectbox('Symmetry Rating:', ['Ideal', 'Excellent', 'Very Good', 'Good', 'Fair', 'Poor'])
fluorescence = st.selectbox('fluorescence Rating:', ['None','Faint','Medium','Strong','Very Slight','Very Strong','Slight','Medium Blue','Strong Blue','Very Strong Blue','Medium White'])

depth = st.number_input('Diamond Depth Percentage:', min_value=0, max_value=100.0, value=0)
width = st.number_input('Diamond Table Percentage:', min_value=0, max_value=100.0, value=0)
x = st.number_input('Diamond Length (X) in mm:', min_value=0, max_value=100.0, value=0)
y = st.number_input('Diamond Width (Y) in mm:', min_value=0, max_value=100.0, value=0)
z = st.number_input('Diamond Height (Z) in mm:', min_value=0, max_value=100.0, value=0)

price_d = [0, 0, 0]
price_s = [0, 0, 0]

if st.button('Predict Price'):

    if depth != 0 and width != 0 and x != 0 and y != 0 and z != 0 and carat!= 0 and cut!= 0 :
        price_d = list(predict_DiamondSE(shape, carat, cut, color, clarity, width, depth, x, y, z))

    price_s = list(predict_StoneAlgo(shape, carat, cut, color, clarity, polish, symmetry, fluorescence))

st.subheader('Results using StoneAlgo data')

col4, col5, col6 = st.columns(3)
col4.metric("Linear Reg", price_s[0], delta=None)
col5.metric("KNN Reg", price_s[1], delta=None)
col6.metric("XGBoost Reg", price_s[2], delta=None)

st.subheader('Results using DiamondSE data')

col1, col2, col3 = st.columns(3)
col1.metric("Linear Reg", price_d[0], delta=None)
col2.metric("KNN Reg", price_d[1], delta=None)
col3.metric("XGBoost Reg", price_d[2], delta=None)
