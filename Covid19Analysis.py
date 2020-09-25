import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# The below excel sheet data is upto may,18
data1=pd.read_csv("case_time_series.csv")

print(data1)

fig = px.line(data1, x = 'Date', y = 'Total Confirmed', title='Covid-19 India')
#fig = px.line(data1, x = 'Date', y = 'Total Deceased', title='Covid-19 India')
fig.show()
#fig1.show()
