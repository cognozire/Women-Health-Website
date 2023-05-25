import pandas as pd
import streamlit as st
from sklearn import preprocessing
from statsmodels.tsa.arima.model import ARIMA as ARIMA
import plotly.express as px


PAGE_CONFIG = {"page_title": "Predict Your Weight",
               "page_icon": "chart_with_upwards_trend:", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://cdn.pixabay.com/photo/2023/02/01/21/40/pink-7761356_640.png");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

def showGraphList():
    graph = ["Prediction"]
    opt = st.radio("Prediction", graph)
    return opt

def sidebar():
    global df1, filename, option, opt, columnList
    df1 = None
    allowedExtension = ['csv', 'xlsx']
    with st.sidebar:
        uploaded_file = st.sidebar.file_uploader(
            label="Upload your data", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            filename = uploaded_file.name
            extension = filename[filename.index(".")+1:]
            filename = filename[:filename.index(".")]

            if extension in allowedExtension:
                df1 = pd.read_csv(uploaded_file)
                columnList = df1.columns.values.tolist()
                # option = st.selectbox("Select Column", columnList)
                # st.subheader("Filters")
                opt = showGraphList()
            else:
                st.write("File Format is not supported")



def mainContent():
    if df1 is not None:
        df_o = df1
        df_o = df_o.drop(['Actual User ID'], axis = 1)
        bool_cols = [col for col in df_o.columns if df_o[col].dtype == 'bool' or df_o[col].dtype =='object']
        label_encoder = preprocessing.LabelEncoder()
        for i in bool_cols[1:]:
            df_o[i] = label_encoder.fit_transform(df_o[i])
        df_o['Date'] = pd.to_datetime(df_o['Date'])
        df_o['date'] = df_o['Date'].dt.date
        df_o = df_o.drop(['Date', 'ID'], axis=1)
        df_o = df_o[['Height', 'Gender', 'Age', 'Regular cycle', 'Irregular cycle',
            'No cycle', 'On birth control', 'Menopause', 'Pcos', 'Endo',
            'Calorie target', 'Protein target', 'Carbs target', 'Fat target',
            'Water intake', 'Calorie', 'Protein', 'Fat', 'Carb', 'Steps',
            'Calorie accuracy', 'Protein accuracy', 'Fat accuracy', 'Carb accuracy',
            'Guessed calories', 'Guessed tracked %', 'Sleep hours',
            'Quality of sleep', 'Stress level', 'How do you feel physically',
            'How do you feel emotionally', 'Fiber', 'Phone before bed',
            'Menstrual cycle day', 'Menstrual flow', 'Menstrual mood', 'Bloating?',
            'Craving?', 'Water retention?', 'date', 'Weight']]
        df_o = df_o.drop(['Height','Gender','Age','Regular cycle', 'No cycle', 'Calorie target', 'On birth control', 'Carb accuracy', 'Protein target', 'Carbs target', 'Fat target','Calorie accuracy'], axis = 1)
        mean_weight_all = df_o['Weight'].mean(skipna=True)
        df_o['Weight'].fillna(mean_weight_all, inplace=True)
        corr = df_o.drop(['date'],axis = 1).corr(method='pearson')
        corr['index'] = corr.index
        df = df_o.iloc[-50:,-2:]
        df = df.set_index('date')
        model = ARIMA(df['Weight'],order = (2,1,2))
        model_fit = model.fit()
        data = pd.DataFrame()
        data["weight"] = model_fit.predict(start=51,end=57,dynamic=True)
        df['date'] = df.index
        a = pd.DataFrame({'date': pd.date_range(start=df.date.iloc[-1], periods=8, freq='d', closed='right')})
        d_final = pd.DataFrame()
        d_final['Date'] = a
        d_final['Date'] = pd.to_datetime(d_final['Date']).dt.date
        col_list = data.weight.values.tolist()
        d_final['Estimated Weight'] = col_list
        
        df_graph = df_o[df_o['Menstrual cycle day'].notnull()].head(20) # for line chart

        if opt == "Prediction":
            st.header("Weight Trend")
            st.write(d_final)
            st.header("Top 5 features contributing to your weight are:")
            
            corr_coeffs = corr.corr()['Weight']
            sorted_coeffs = corr_coeffs.abs().sort_values(ascending=False)
            top_5_features = sorted_coeffs.index[1:6]
            for i in top_5_features:
                st.write(i)
            st.header("Relationship between Weight and Menstrual Cycle Day")
            fig = px.line(df_graph, x = "Menstrual cycle day",y = "Weight")
            st.plotly_chart(fig)
#         elif opt =="Slider":
            
#             corr_coeffs = corr.corr()['Weight']
#             corr_coeffs = corr_coeffs.to_frame()

            
#             w = st.number_input("Enter your weight here (lbs):")
            
#             stress_cor = corr_coeffs.at["Stress level", "Weight"]
#             step_cor = corr_coeffs.at["Steps", "Weight"]
#             sleep_cor = corr_coeffs.at["Sleep hours", "Weight"]
#             calorie_cor = corr_coeffs.at["Calorie", "Weight"]
            
#             stress = st.slider('Stress Level (%)', 0, 100, 0)
# #             w1 = w+w*((stress_cor+(stress*0.001)))
#             del_w1 = stress_cor*w
#             w_adj1 = (stress-50)/50
#             w1 = w + (w_adj1*del_w1)
#             st.write("New Weight : ", 0 if stress == 0 else w1)
  
#             steps = st.slider('Steps (%)', 0, 100, 0)
# #             w2 = w+w*((step_cor+(Steps*0.01)))
#             del_w2 = step_cor*w
#             w_adj2 = (steps-50)/50
#             w2 = w + (w_adj2*del_w2)
#             st.write("New Weight : ", 0 if steps == 0 else w2)
  
#             sleep = st.slider('Sleep (%)', 0, 100, 0)
# #             w3 = w+w*((sleep_cor+(sleep*0.01)))
#             del_w3 = sleep_cor*w
#             w_adj3 = (sleep-50)/50
#             w3 = w + (w_adj3*del_w3)
#             st.write("New Weight : ", 0 if sleep == 0 else w3)
  
#             calorie = st.slider('Calorie (%)', 0, 100, 0)
# #             w4 = w+w*((calorie_cor+(Calorie*0.01)))
#             del_w4 = calorie_cor*w
#             w_adj4 = (calorie-50)/50
#             w4 = w + (w_adj4*del_w4)
#             st.write("New Weight : ", 0 if calorie == 0 else w4)
            
        else:
            st.write("There is nothing to show!! Please add file to see data.")

if __name__ == "__main__":
    footer = """
    <div style='position: fixed; bottom: 0; width: 100%; text-align: center; font-size: 12px;'>
        <p>Made with ❤️ by Cognozire</p>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)
    sidebar()
    st.markdown("<h1 style='text-align: center; color: black;'>Women Daily Health Check</h1>", unsafe_allow_html=True)
    mainContent()
    
