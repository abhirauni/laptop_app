import streamlit as st
import joblib
import numpy as np






# import the model
pipe = joblib.load(open('pipe4.pkl','rb'))
df = joblib.load(open('df4.pkl','rb'))


st.title("LAPTOP PREDICTOR")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the lap')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])


# Ips
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.number_input("Screen size")

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])


# cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())
# hdd
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
# ssd
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
# gpu
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())
# os
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res*2) + (Y_res))*0.5/(screen_size+1)
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)

    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
