
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report
# from mplfinance import candlestick_ohlc



def Welcome():
    return "WELCOME USERS"

def inputs(PrevClose,Low,VWAP,Volume,DeliverableVolume,Deliverble):
    prediction=model.predict([[PrevClose,Low,VWAP,Volume,DeliverableVolume,Deliverble]])
    print(prediction)
    return prediction

def adani_inputs(Open,High,Low,AdjClose,Volume):
    prediction1=model.predict([[Open,High,Low,AdjClose,Volume]])
    print(prediction1)
    return prediction1

comp=st.sidebar.selectbox("Company",["Nestle","Adani","Maruti","NestleInd","JSW Steel","Reliance","CoalInd","Cipla","Britania"])
nav=st.sidebar.radio("Navigator",["Home","Predictor","Graphology","FAQ"])
visulizer=st.sidebar.selectbox("StockGrade Visualizer",["None","Candlestick","BoxPlot"])






if comp=="Nestle":
    nestle = open("Nestle_predictor.pkl", "rb")
    model = pickle.load(nestle)
    data = pd.read_csv("C://Users//User//Desktop//Capstone 2//NESTLEIND.csv")

    corr_mat = data.corr()

    if nav == "Home":

        st.markdown("Home Page")
        # st.image("logo.jpeg", width=300)
        st.title("Welcome to StockGrade")
        st.header("Your Trusted bank")
        # st.image("kot.jpg", width=500)

        st.markdown("Want to know the Chart visually?")

        if st.button("Yes"):
            st.line_chart(data)
        elif st.button("No"):
            st.header("Know Our Growth History Below!")
        if st.checkbox("Wanna see our complete dataset?"):
            st.write(data)

        if st.button("Want To know the Growth History Column Wise?"):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,
                                                         ncols=2,
                                                         figsize=(10, 5))
            ax1.bar(data['High'],data['Close'])
            ax2.scatter(data['Low'],data['Close'],c='lightblue')
            ax3.plot(data['Deliverable Volume'],data['Close'])
            ax4.scatter(data['%Deliverble'],data['Close'],c='salmon')
            st.pyplot(fig)

    if nav == "Predictor":

        if st.checkbox("StockGrade Inputs"):
            st.write("1. PrevClose = Previous Closing Price of the company")
            st.write("2. Low = Takes Low price of the stock till date")
            st.write("3. VWAP = volume weighted average price (VWAP)")
            st.write("4. Volume = Amount of the stock")
            st.write("5. DeliverableVolume = Amount to be delivered")
            st.write("6. Deliverble = Original Volume")

        st.title("Welcome To Nestle Predictor!!")
        PrevClose = st.number_input("Enter the Previous Close Price", 27.00, 10000000000.0, step=40.0)
        Low = st.number_input("Enter the Low Price", 27.00, 200000000.0, step=40.0)
        VWAP = st.number_input("Enter VWAP", 27.00, 200000000.0, step=40.0)
        Volume = st.number_input("Enter the Volume", 40.00, 300000000.0, step=200.0)
        DeliverableVolume = st.number_input("Enter Deliverable Volume", 40.0, 100000000.0, step=200.0)
        Deliverble = st.number_input("Enter the Deliverble Value", 0.001,100000000.0, step=10.0)

        result = ""

        if st.button("Predict"):
            result = inputs(PrevClose, Low, VWAP, Volume, DeliverableVolume, Deliverble)
        st.success(f"Your Predicted Price is : {result}")

    if nav == "Graphology":
        st.title("Welcome to StockGrade Graph mate")

        graph = st.selectbox("What Kind Of Graph?",
                             ["None", "Previous Close Price Vs Close Price", "Highest Price Vs Close Price",
                              "Volume Vs Close Price", "Analyze The Complete data at one go","StockGrade Grapher"])

        if graph == "Previous Close Price Vs Close Price":
            sns.set_style("darkgrid")
            prevclo = sns.relplot(data=data, x='Prev Close', y='Close')
            st.pyplot(prevclo)

        if graph == "Highest Price Vs Close Price":
            sns.set_style("darkgrid")
            higclo = sns.relplot(data=data, x='High', y='Close')
            st.pyplot(higclo)
        if graph == "Volume Vs Close Price":
            sns.set_style("darkgrid")
            volvsclo = sns.relplot(data=data, x='Volume', y='Close')
            st.pyplot(volvsclo)
        if graph == "Analyze The Complete data at one go":
            # fig,ax=plt.subplots(figsize=(20,20))
            analyzer = sns.pairplot(corr_mat)
            st.pyplot(analyzer)
            analyzer.fig.set_size_inches(20, 20)

        if graph=="StockGrade Grapher":
            plt.style.use('dark_background')
            fig,ax=plt.subplots(figsize=(10,8))
            ax.scatter(data['Low'],data['High'],c=data['Close'],cmap='winter')
            ax.set(title="Data Analyzer",
                   xlabel="Low",
                   ylabel="High")
            ax.legend()
            ax.axhline(data['High'].mean(),
                       linestyle='--',
                       c='green')
            st.pyplot(fig)
    if nav=='FAQ':
        st.title("Welocme To StockGrade FAQ section")
        st.header("FAQ")
        if st.button("Is StockGrade Safe?"):
            st.write("Now Stock Grade is Predicting the closing prices on the previous old data sets "
                     "So, it isn't predicting the accurate closing prices of the company.Which may also "
                     "make The user fall into loss or in Profit there is no surrity "
                     "As soon as we start getting the live or original data from the companies we ",
                     "asure that StockGrade will lift you upp with profit")

        if st.button("Is stockgrade recommended for real life investment purposes?"):
            st.write("No,As StockGrade has no original data available with them so it is not at all recomended "
                     "for real life predictions.But,StockGrade definitely helps to maske user learn "
                     "What are Predictions? "
                     "How it works "
                     "Necessity in Real Life "
                     "And Much More")

        if st.button("What is the target/ambition of StockGrade?"):

            st.write("StockGrade is an Indian web application. "
                     "Which wants it's every citizen to be good financially good. "
                     "Though StockGrade is not predicting on the real data. "
                     "But StockGrade assures that everyone can understand the importance and use of StockGrade")

        if st.button("Why should you consider investing in stocks?"):

            st.header("1. It’s easy")
            st.write("Investing in stocks has never been so easy."
                     "Now you can invest from the comfort of your homes." 
                     "All you need is a smartphone and you are good to go.")

            st.header( "2. Power of compounding")
            st.write("If you let your investments stay for a long time and let the "
                     " interests compound, you will reap good results and will get one of the best benefits of investing in stocks.")
            st.header("3. Win the race against inflation ")
            st.write("The interests in conventional bank system at times are close to the inflation rates"
                     "leaving you with little or no profit at all in the long term. Stock investment returns can "
                     "fetch you double-digit inflation returns if done intelligently and help you reach the"
                     "corpus you desire in a relatively shorter time frame.")
            st.header("5. The powerful long term investment ")
            st.write("Bajaj Finance, a non-banking finance company, between December 2009 and December "
                     "2019 gave a whopping 13,000% returns in its stock. However, this does not mean that"
                     "every investment can yield returns so high but it will certainly serve as a great tool to "
                     "multiply your money to the best extent possible")

if comp=='Adani':
    adani = open("Adani_predictor.pkl", "rb")
    model = pickle.load(adani)
    data = pd.read_csv("C://Users//User//Desktop//Capstone 2//ADANIPORTS.NS.csv")

    corr_mat = data.corr()

    if nav == "Home":

        st.markdown("Home Page")
        # st.image("logo.jpeg", width=300)
        st.title("Welcome to StockGrade")
        st.header("Your Trusted bank")
        # st.image("kot.jpg", width=500)

        st.markdown("Want to know the Chart visually?")

        if st.button("Yes"):
            st.line_chart(data)
        elif st.button("No"):
            st.header("Know Our Growth History Below!")
        if st.checkbox("Wanna see our complete dataset?"):
            st.write(data)

        if st.button("Want To know the Growth History Column Wise?"):
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2,
                                                         ncols=2,
                                                         figsize=(10, 5))
            ax1.bar(data['High'], data['Close'])
            ax2.scatter(data['Low'], data['Close'], c='lightblue')
            ax3.plot(data['High'], data['Close'])
            ax4.scatter(data['Adj Close'], data['Close'], c='salmon')
            st.pyplot(fig)
    if nav == "Predictor":

        if st.checkbox("StockGrade Inputs"):
            st.write("1. PrevClose = Previous Closing Price of the company")
            st.write("2. Low = Takes Low price of the stock till date")
            st.write("3. VWAP = volume weighted average price (VWAP)")
            st.write("4. Volume = Amount of the stock")
            st.write("5. DeliverableVolume = Amount to be delivered")
            st.write("6. Deliverble = Original Volume")

        st.title("Welcome To Adani Predictor!!")
        Open = st.number_input("Enter the Open Price", 27.00, 1000000.00, step=40.0)
        High = st.number_input("Enter the Highest Price of the day", 27.00, 1000000.00, step=40.0)
        Low = st.number_input("Enter the Low price", 27.00, 2038.00, step=40.0)
        AdjClose = st.number_input("Enter the Adjacent Close Price", 40.00, 10000000.0, step=200.0)
        Volume = st.number_input("Enter the Volume", 40.0, 1000000000000.00, step=200.0)

        result = ""

        if st.button("Predict"):
            result = adani_inputs(Open,High,Low,AdjClose,Volume)
        st.success(f"Your Predicted Price is : {result}")

    if nav == "Graphology":
        st.title("Welcome to StockGrade Graph mate")

        graph = st.selectbox("What Kind Of Graph?",
                             ["None", "Previous Close Price Vs Close Price", "Highest Price Vs Close Price",
                              "Volume Vs Close Price", "Analyze The Complete data at one go", "StockGrade Grapher"])

        if graph == "Previous Close Price Vs Close Price":
            sns.set_style("darkgrid")
            prevclo = sns.relplot(data=data, x='Adj Close', y='Close')
            st.pyplot(prevclo)

        if graph == "Highest Price Vs Close Price":
            sns.set_style("darkgrid")
            higclo = sns.relplot(data=data, x='High', y='Close')
            st.pyplot(higclo)
        if graph == "Volume Vs Close Price":
            sns.set_style("darkgrid")
            volvsclo = sns.relplot(data=data, x='Volume', y='Close')
            st.pyplot(volvsclo)
        if graph == "Analyze The Complete data at one go":
            # fig,ax=plt.subplots(figsize=(20,20))
            analyzer = sns.pairplot(corr_mat)
            st.pyplot(analyzer)
            analyzer.fig.set_size_inches(20, 20)

        if graph == "StockGrade Grapher":
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.scatter(data['Low'], data['High'], c=data['Close'], cmap='winter')
            ax.set(title="Data Analyzer",
                   xlabel="Low",
                   ylabel="High")
            ax.legend()
            ax.axhline(data['High'].mean(),
                       linestyle='--',
                       c='green')
            st.pyplot(fig)


    if nav=='FAQ':
        st.title("Welocme To StockGrade FAQ section")
        st.header("FAQ")
        if st.button("Is StockGrade Safe?"):
            st.write("Now Stock Grade is Predicting the closing prices on the previous old data sets "
                     "So, it isn't predicting the accurate closing prices of the company.Which may also "
                     "make The user fall into loss or in Profit there is no surrity "
                     "As soon as we start getting the live or original data from the companies we ",
                     "asure that StockGrade will lift you upp with profit")

        if st.button("Is stockgrade recommended for real life investment purposes?"):
            st.write("No,As StockGrade has no original data available with them so it is not at all recomended "
                     "for real life predictions.But,StockGrade definitely helps to maske user learn "
                     "What are Predictions? "
                     "How it works "
                     "Necessity in Real Life "
                     "And Much More")

        if st.button("What is the target/ambition of StockGrade?"):

            st.write("StockGrade is an Indian web application. "
                     "Which wants it's every citizen to be good financially good. "
                     "Though StockGrade is not predicting on the real data. "
                     "But StockGrade assures that everyone can understand the importance and use of StockGrade")

        if st.button("Why should you consider investing in stocks?"):

            st.header("1. It’s easy")
            st.write("Investing in stocks has never been so easy."
                     "Now you can invest from the comfort of your homes." 
                     "All you need is a smartphone and you are good to go.")

            st.header( "2. Power of compounding")
            st.write("If you let your investments stay for a long time and let the "
                     " interests compound, you will reap good results and will get one of the best benefits of investing in stocks.")
            st.header("3. Win the race against inflation ")
            st.write("The interests in conventional bank system at times are close to the inflation rates"
                     "leaving you with little or no profit at all in the long term. Stock investment returns can "
                     "fetch you double-digit inflation returns if done intelligently and help you reach the"
                     "corpus you desire in a relatively shorter time frame.")
            st.header("5. The powerful long term investment ")
            st.write("Bajaj Finance, a non-banking finance company, between December 2009 and December "
                     "2019 gave a whopping 13,000% returns in its stock. However, this does not mean that"
                     "every investment can yield returns so high but it will certainly serve as a great tool to "
                     "multiply your money to the best extent possible")
