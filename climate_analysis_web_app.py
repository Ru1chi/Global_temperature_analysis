
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 19:14:07 2024

@author: sr322
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.simplefilter("ignore")
from PIL import Image

# Load Data
@st.cache_data
def load_data():
    Global_Temperature = pd.read_csv("GlobalTemperatures.csv")
    Temperature_Country = pd.read_csv("GlobalLandTemperaturesByCountry.csv")
    Temperature_City = pd.read_csv("GlobalLandTemperaturesByCity.csv")
    return Global_Temperature, Temperature_Country, Temperature_City

Global_Temperature, Temperature_Country, Temperature_City = load_data()

# Data Cleaning and Preparation
Temperature_City.dropna(axis= 0 , subset= ['AverageTemperature'] , inplace= True)
Temperature_Country.dropna(axis= 0 , subset=  ['AverageTemperature'] , inplace= True)

Global_Temperature['Year'] = pd.to_datetime(Global_Temperature['dt']).dt.year
Temperature_City['Year'] = pd.to_datetime(Temperature_City['dt']).dt.year
Temperature_Country['Year'] = pd.to_datetime(Temperature_Country['dt']).dt.year

Global_Temperature['Month'] = pd.to_datetime(Global_Temperature['dt']).dt.month
Temperature_City['Month'] = pd.to_datetime(Temperature_City['dt']).dt.month
Temperature_Country['Month'] = pd.to_datetime(Temperature_Country['dt']).dt.month

def season_name(month_number):
    seasons = {
        1: 'Winter',
        2: 'Winter',
        3: 'Spring',
        4: 'Spring',
        5: 'Spring',
        6: 'Summer',
        7: 'Summer',
        8: 'Summer',
        9: 'Autumn',
        10: 'Autumn',
        11: 'Autumn',
        12: 'Winter'
    }
    return seasons.get(month_number)
Global_Temperature['Season'] = Global_Temperature['Month'].apply(season_name)

def get_month_name(month_number):
    months = {
        1: 'January',
        2: 'February',
        3: 'March',
        4: 'April',
        5: 'May',
        6: 'June',
        7: 'July',
        8: 'August',
        9: 'September',
        10: 'October',
        11: 'November',
        12: 'December'
    }
    return months.get(month_number)
Global_Temperature['Month_Name'] = Global_Temperature['Month'].apply(get_month_name)

def get_century(year):
    century = (year - 1) // 100 + 1
    return century
Global_Temperature['Century'] = Global_Temperature['Year'].apply(get_century)
Temperature_City['Century'] = Temperature_City['Year'].apply(get_century)
Temperature_Country['Century'] = Temperature_Country['Year'].apply(get_century)

Global_Temperature['OceanAverageTemperature'] = Global_Temperature['LandAndOceanAverageTemperature'] - Global_Temperature['LandAverageTemperature']

def lat_cor(x):
    if isinstance(x, str):
        if x[-1] == 'N' :
            return float(x[:-1])
        elif x[-1] == 'S' :
            return float("-" + x[:-1])
    return None
Temperature_City['Lat_Cor'] = Temperature_City['Latitude'].apply(lat_cor)

def lon_cor(x):
    if isinstance(x, str):
        if x[-1] == 'E':
            return float(x[:-1])
        elif x[-1] == 'W':
            return -float(x[:-1])
    return None
Temperature_City['Lon_Cor'] = Temperature_City['Longitude'].apply(lon_cor)

# Streamlit App Layout
st.title("Global Temperature Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Overview", "Data Info", "Visualizations", "Forecasting", "Conclusion"]
choice = st.sidebar.selectbox("Go to", options)

if choice == "Overview":
    st.subheader("Overview of the Project")
    st.write("""
    This project analyzes global temperature data over several centuries, focusing on land and ocean temperatures. 
    The analysis covers seasonal trends, temperature changes over centuries, and a prediction of future temperatures.
    """)
    image = Image.open("C:/Users/sr322/OneDrive/Desktop/temp.jpg")  
    st.image(image, use_column_width=True)


elif choice == "Data Info":
    st.subheader("Dataset Information")
    st.write("### Global Temperature Dataset")
    st.write(Global_Temperature.describe())
    st.write("### Temperature by Country Dataset")
    st.write(Temperature_Country.describe())
    st.write("### Temperature by City Dataset")
    st.write(Temperature_City.describe())

elif choice == "Visualizations":
    st.subheader("Data Visualizations")

    # Plot 1: Line Plot
    st.markdown("### Average Land Temperature and Average Land Temperature Uncertainty Trend Over the Years")
    fig1 = px.line(data_frame=Global_Temperature.groupby(['Year', 'Month'])[['LandAverageTemperature','LandAverageTemperatureUncertainty']].mean().reset_index(),
            x='Year', y=['LandAverageTemperature', 'LandAverageTemperatureUncertainty'])
    fig1.update_layout(yaxis=dict(title_text='Temperature'), height=500, width=950)
    st.plotly_chart(fig1)

    # Plot 2: Line Plot Comparison
    st.markdown("### Comparison between Ocean and Land Average Temperature")
    fig2, ax = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=Global_Temperature.dropna(subset='OceanAverageTemperature'), y='LandAverageTemperature', x='Year', label='Land Average Temperature', color='orange', ax=ax)
    sns.lineplot(data=Global_Temperature.dropna(subset='OceanAverageTemperature'), y='OceanAverageTemperature', x='Year', label='Ocean Average Temperature', color='blue', ax=ax)
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature')
    ax.legend()
    sns.set_style("whitegrid")
    st.pyplot(fig2)

    # Plot 3: Scatter Plot
    st.markdown("### Scatter Plot of Land and Ocean Temperature")
    fig3, ax = plt.subplots(figsize=(8, 5))
    data_to_plot = Global_Temperature.groupby('Year')[['LandAverageTemperature', 'OceanAverageTemperature']].mean().reset_index()
    sns.scatterplot(data=data_to_plot, x='Year', y='LandAverageTemperature', label='Land Average Temperature', ax=ax)
    sns.scatterplot(data=data_to_plot, x='Year', y='OceanAverageTemperature', label='Ocean Average Temperature', ax=ax)
    sns.regplot(data=data_to_plot, x='Year', y='LandAverageTemperature', scatter=False, color='red', ax=ax)
    sns.regplot(data=data_to_plot, x='Year', y='OceanAverageTemperature', scatter=False, color='red', ax=ax)
    ax.set_ylabel("Temperature")
    st.pyplot(fig3)

    # Plot 4: KDE Plots
    st.markdown("### KDE Plots")
    fig4, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.kdeplot(data=Global_Temperature, x='Year', y='LandAverageTemperature', ax=axs[0])
    axs[0].set_ylabel("Temperature")
    axs[0].set_title("Isotherm Map by Years")
    
    sns.kdeplot(data=Global_Temperature, x='LandAverageTemperature', ax=axs[1])
    axs[1].set_xlabel("Temperature")
    axs[1].set_title('Temperature Frequency Graph')
    
    sns.set_style("whitegrid")
    st.pyplot(fig4)

    # Plot 5: Choropleth
    st.markdown("### Average Temperature in 2013 by Countries")
    fig5 = px.choropleth(Temperature_Country.loc[Temperature_Country['Year'] == 2013, :].groupby('Country')['AverageTemperature'].mean().reset_index(),
                        locations='Country', height=500, width=850,
                        locationmode='country names', color='AverageTemperature',
                        hover_name='Country',
                        color_continuous_scale='RdBu_r')
    fig5.update_layout(title_x=0.5)
    st.plotly_chart(fig5)

    # Plot 6: Contour Plot
    st.markdown("### Filled Contour Plot of Average Temperature")
    fig6, ax = plt.subplots(figsize=[6, 4])
    contour = plt.tricontourf(Temperature_City['Lon_Cor'], Temperature_City['Lat_Cor'], Temperature_City['AverageTemperature'], cmap='YlOrRd')
    plt.colorbar(contour, ax=ax, label='Average Temperature')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    st.pyplot(fig6)

    # Plot 7: Box Plot by Century
    st.markdown("### Box Plot by Century")
    fig7 = px.box(Global_Temperature, y='LandAverageTemperature', x='Century', color='Century', orientation='v')
    st.plotly_chart(fig7)

    # Plot 8: Land Average Temperature by Season
    st.markdown("### Land Average Temperature by Season")
    fig8 = px.box(Global_Temperature, y='LandAverageTemperature', x='Season', color='Season')
    fig8.update_layout(yaxis_title="Temperature", height=500, width=800)
    st.plotly_chart(fig8)

    # Plot 9: Vertical Box Plot by Month
    st.markdown("### Land Average Temperature by Month")
    fig9 = px.box(Global_Temperature, y='LandAverageTemperature', x='Month_Name', color='Month_Name', orientation='v')
    fig9.update_layout(yaxis_title="Temperature", height=500, width=800)
    st.plotly_chart(fig9)

    # Plot 10: Top 5 and Bottom 5 Countries by Average Temperature in 21st Century
    st.markdown("### Bottom 5 Countries by Average Temperature in 21st Century")
    temp_by_country_century = Temperature_Country[Temperature_Country['Century'] == 21].groupby('Country')['AverageTemperature'].mean().sort_values().reset_index()
    fig10_1 = px.bar(temp_by_country_century.head(5), x='Country', y='AverageTemperature', color='Country')
    fig10_1.update_layout(yaxis_title="Temperature", height=400, width=800)
    st.plotly_chart(fig10_1)

    st.markdown("### Top 5 Countries by Average Temperature in 21st Century")
    fig10_2 = px.bar(temp_by_country_century.tail(5), x='Country', y='AverageTemperature', color='Country')
    fig10_2.update_layout(yaxis_title="Temperature", height=400, width=800)
    st.plotly_chart(fig10_2)

    # Plot 11: Temperature Trend for Top 5 and Bottom 5 Countries
    st.markdown("### Temperature Trend for Top 5 Countries")
    top_5_countries = temp_by_country_century.tail(5)['Country'].tolist()
    fig11_1 = px.line(Temperature_Country[Temperature_Country['Country'].isin(top_5_countries)].groupby(['Year', 'Country'])['AverageTemperature'].mean().reset_index(),
                      x='Year', y='AverageTemperature', color='Country')
    fig11_1.update_layout(yaxis_title="Temperature", height=500, width=800)
    st.plotly_chart(fig11_1)

    st.markdown("### Temperature Trend for Bottom 5 Countries")
    bottom_5_countries = temp_by_country_century.head(5)['Country'].tolist()
    fig11_2 = px.line(Temperature_Country[Temperature_Country['Country'].isin(bottom_5_countries)].groupby(['Year', 'Country'])['AverageTemperature'].mean().reset_index(),
                      x='Year', y='AverageTemperature', color='Country')
    fig11_2.update_layout(yaxis_title="Temperature", height=500, width=800)
    st.plotly_chart(fig11_2)


elif choice == "Forecasting":
    st.subheader("Temperature Forecasting")

    # Forecasting using ARIMA
    temperature_series = Global_Temperature.groupby('Year')['LandAverageTemperature'].mean()
    model = ARIMA(temperature_series, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=12)
    forecast_years = np.arange(temperature_series.index[-1] + 1, temperature_series.index[-1] + len(forecast) + 1)
    
    # Plotting the forecast
    fig12, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temperature_series.index, temperature_series, label='Historical Data')
    ax.plot(forecast_years, forecast, label='Forecast', color='red')
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Forecast for Next Year')
    ax.legend()
    st.pyplot(fig12)

elif choice == "Conclusion":
    st.subheader("Conclusion")
    st.write("""
    The analysis shows a significant increase in global temperatures over the centuries, with noticeable trends in both land and ocean temperatures. 
    The top 5 countries with the highest temperatures and the bottom 5 with the lowest temperatures exhibit distinct patterns over time. 
    The forecast indicates a continued rise in temperature, emphasizing the ongoing impact of climate change.
    """)
    image = Image.open("C:/Users/sr322/OneDrive/Desktop/temp2.jpg") 
    st.image(image, use_column_width=True)


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
