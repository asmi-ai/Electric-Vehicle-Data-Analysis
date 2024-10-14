#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# In[42]:


dataset = pd.read_csv('dataset.csv')


# In[43]:


dataset.info()


# In[44]:


dataset.shape


# In[45]:


dataset.columns


# In[46]:


dataset.describe()


# In[47]:


# Separate categorical and numerical columns
categorical_columns = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_columns = dataset.select_dtypes(include=['number']).columns.tolist()

# Print the lists of columns
print("Categorical Columns:", categorical_columns)
print("Numerical Columns:", numerical_columns)


# In[48]:


dataset.isnull().sum()


# ### Impute missing values
# 

# In[49]:


# 1. Impute 'Model' with the most frequent value (mode)
dataset['Model'].fillna(dataset['Model'].mode()[0], inplace=True)


# In[50]:


# 2. Impute 'Legislative District' with the median
dataset['Legislative District'].fillna(dataset['Legislative District'].median(), inplace=True)


# In[51]:


# 3. Impute 'Vehicle Location' with 'Unknown'
dataset['Vehicle Location'].fillna('Unknown', inplace=True)


# In[52]:


# 4. Impute 'Electric Utility' with the most frequent value (mode)
dataset['Electric Utility'].fillna(dataset['Electric Utility'].mode()[0], inplace=True)


# In[53]:


# Check the missing values count after imputation
dataset.isnull().sum()


# In[54]:


dataset.duplicated().sum()


# ### DATA VISUALIZATION

# #### UNIVARIATE ANALYSIS- Numerical variables

# ##### 1. model year

# In[55]:


fig_hist_model_year = px.histogram(dataset, x='Model Year', title='Histogram of Model Year', nbins=20)
fig_hist_model_year.show()


# The histogram of Model Year shows a significant increase in the number of electric vehicles (EVs) produced after 2010, with a sharp rise between 2015 and 2020. This suggests that EV production has been growing rapidly in recent years, with 2020 having the highest count in the dataset. The earlier years have a relatively lower count, indicating that EV adoption has significantly accelerated in the past decade.

# In[56]:


fig_box_model_year = px.box(dataset, y='Model Year', title='Box Plot of Model Year')
fig_box_model_year.show()


# The Box Plot of Model Year shows that the majority of electric vehicles in the dataset were produced between 2015 and 2020, with a median close to 2020. There are several outliers from earlier years, particularly from 2000 to 2010, indicating that there were some EVs produced during that period but in relatively smaller numbers. The box plot suggests that most EVs in the dataset are from recent years, further supporting the trend of rapid growth in EV production after 2015.

# In[57]:


summary_model_year = dataset['Model Year'].describe()
print("Summary statistics for Model Year:")
print(summary_model_year)


# ##### 2. electric range

# In[58]:


fig_hist_electric_range = px.histogram(dataset, x='Electric Range', title='Histogram of Electric Range', nbins=20)
fig_hist_electric_range.show()


# The Histogram of Electric Range shows that most electric vehicles have a range of under 50 miles, with significant counts also around 100 and 200 miles. The distribution indicates a long tail with few vehicles exceeding 200 miles and very few beyond 300 miles. This highlights a potential area for improvement in future EV models to meet the demand for longer ranges.

# In[59]:


fig_box_electric_range = px.box(dataset, y='Electric Range', title='Box Plot of Electric Range')
fig_box_electric_range.show()


# In[60]:


summary_electric_range = dataset['Electric Range'].describe()
print("Summary statistics for Electric Range:")
print(summary_electric_range)


# ##### 3. base MSRP

# In[61]:


fig_hist_base_msrp = px.histogram(dataset, x='Base MSRP', title='Histogram of Base MSRP', nbins=20)
fig_hist_base_msrp.show()


# In[62]:


fig_box_base_msrp = px.box(dataset, y='Base MSRP', title='Box Plot of Base MSRP')
fig_box_base_msrp.show()


# In[63]:


summary_base_msrp = dataset['Base MSRP'].describe()
print("Summary statistics for Base MSRP:")
print(summary_base_msrp)


# In[64]:


# Get value counts for each categorical column
for col in categorical_columns:
    print(f"Value counts for {col}:")
    print(dataset[col].value_counts())
    print("\n")


# #### UNIVARIATE ANLAYSIS - CATEGORICAL VARIABLES

# In[65]:


### Univariate Analysis for Categorical Variables ###
for col in categorical_columns:
    # Bar Plot for categorical variables
    fig = px.bar(dataset[col].value_counts().reset_index(), x='index', y=col, 
                 title=f'Count Plot of {col}', 
                 labels={'index': col, col: 'Count'},
                 color_discrete_sequence=['green'])
    fig.update_layout(xaxis_title=col, yaxis_title='Count')
    fig.show()

    # Print frequency of categories for each categorical column
    print(f"Frequency of categories for {col}:")
    print(dataset[col].value_counts())
    print("\n")


# #### BIVARIATE ANALYSIS 

# In[66]:


fig_scatter_range_msrp = px.scatter(dataset, x='Electric Range', y='Base MSRP', title='Electric Range vs. Base MSRP')
fig_scatter_range_msrp.show()


# In[67]:


fig_scatter_ev_range_model_year = px.scatter(dataset, x='Model Year', y='Electric Range', title='Electric Range vs. Model Year')
fig_scatter_ev_range_model_year.show()


# In[68]:


fig_scatter_model_year_ev_count = px.scatter(dataset['Model Year'].value_counts(), title='Number of Electric Vehicles by Model Year')
fig_scatter_model_year_ev_count.show()


# The plot shows an upward trend in the index values starting from around 2010, with sharp increases after 2015, reaching over 25k by the most recent model years. This suggests a strong increase in the variable (likely Model Year) as time progresses.

# In[69]:


# Scatter plot for Model Year vs Base MSRP
fig_2 = px.scatter(dataset, x='Model Year', y='Base MSRP',
                   title='Scatter Plot of Model Year vs Base MSRP',
                   labels={'Model Year': 'Model Year', 'Base MSRP': 'Base MSRP'})
fig_2.show()


# The plot shows that most vehicles from 2010-2020 have a Base MSRP below 200k, with a few outliers, including one above 800k in 2015. Older models (pre-2005) tend to have much lower prices, and there’s no clear trend linking model year to price.

# In[70]:


# Box plot for Electric Vehicle Type vs Base MSRP
fig = px.box(dataset, x='Electric Vehicle Type', y='Base MSRP',
             title='Electric Vehicle Type vs Base MSRP',
             labels={'Electric Vehicle Type': 'Electric Vehicle Type', 'Base MSRP': 'Base MSRP'})
fig.show()


# In[71]:


fig2 = px.box(dataset, x='Electric Vehicle Type', y='Electric Range', 
              title='Electric Vehicle Type vs. Electric Range',
              labels={''})
fig2.show()


# In[72]:


# Box plot for State vs Electric Range
fig3 = px.box(dataset, x='State', y='Electric Range',
               title='State vs Electric Range',
               labels={'State': 'State', 'Electric Range': 'Electric Range (miles)'})
fig3.show()


# In[73]:


fig_box_ev_type = px.violin(dataset, x='Electric Vehicle Type', y='Electric Range', title='Electric Vehicle Type vs. Electric Range')
fig_box_ev_type.show()


# In[74]:


fig_box = px.box(dataset, x='Clean Alternative Fuel Vehicle (CAFV) Eligibility', y='Electric Range', 
                 title='Clean Alternative Fuel Vehicle (CAFV) Eligibility vs. Electric Range',
                 labels={'Clean Alternative Fuel Vehicle (CAFV) Eligibility': 'CAFV Eligibility', 
                         'Electric Range': 'Electric Range (miles)'})

fig_box.show()


# ##### CORRELATION MATRIX

# In[75]:


# Step 1: Calculate the correlation matrix
corr_matrix = dataset.corr()

# Step 2: Plot the correlation matrix using Plotly Express
fig = px.imshow(corr_matrix, 
                labels=dict(color="Correlation"),
                x=corr_matrix.columns, 
                y=corr_matrix.columns,
                title="Correlation Heatmap",
                color_continuous_scale='RdBu_r', 
                zmin=-1, zmax=1)
fig.update_layout(xaxis_title="Features", yaxis_title="Features")
fig.show()


# Correlation Strength:
# 
# Dark red colors indicate a strong positive correlation (close to 1).
# Dark blue colors indicate a strong negative correlation (close to -1).
# White/neutral colors indicate little to no correlation (around 0).
# Key Observations:
# 
# There is a strong positive correlation between Base MSRP (Manufacturer's Suggested Retail Price) and Electric Range, meaning vehicles with a higher MSRP tend to have longer electric ranges.
# Legislative District and DOL Vehicle ID are highly correlated with each other and possibly with other features.
# Features such as Postal Code and 2020 Census Tract seem to have less correlation with other features, suggesting that these may not directly influence the variables considered.

# # Task 2: Create a Choropleth using plotly.express to display the number of EV vehicles based on location.

# In[76]:


# Function to extract Longitude and Latitude
def extract_coordinates(loc):
    try:
        parts = loc.split()
        longitude = float(parts[1][1:])  # Assumes the second part is "lon:xxxx"
        latitude = float(parts[2][:-1])   # Assumes the third part is "lat:xxxx"
        return longitude, latitude
    except (IndexError, ValueError):
        return None, None  # Return None if there's an error

# Apply the function to extract coordinates
dataset['Longitude'], dataset['Latitude'] = zip(*dataset['Vehicle Location'].apply(extract_coordinates))

# Drop rows where Longitude or Latitude is None
dataset = dataset.dropna(subset=['Longitude', 'Latitude'])

# Group by Latitude, Longitude, Postal Code, County, and State to count EVs
location_counts = dataset.groupby(['Latitude', 'Longitude', 'Postal Code', 'County', 'State']).size().reset_index(name='EV Count')

# Create the scatter map
fig_scatter_map = px.scatter_mapbox(location_counts,
                                     lat='Latitude',
                                     lon='Longitude',
                                     color='EV Count',
                                     size='EV Count',
                                     mapbox_style='carto-positron',
                                     zoom=3,
                                     center={'lat': 37.0902, 'lon': -95.7129},
                                     title='Scatter Map of Electric Vehicle Locations')

# Show the map
fig_scatter_map.show()


# In[77]:


import imageio
import os
from IPython.display import Image, display


# In[78]:


# Save the figure as PNG
if not os.path.exists('scatter_map_images'):
    os.makedirs('scatter_map_images')

for i in range(5):  # Adjust based on how many frames you want
    fig_scatter_map.update_layout(mapbox_zoom=3 + i)  # Change zoom or other parameters per frame
    fig_scatter_map.write_image(f'scatter_map_images/scatter_map_frame_{i}.png', engine="kaleido")

# Convert saved PNGs to GIF
images = []
for i in range(5):  # Adjust to match the number of frames
    images.append(imageio.imread(f'scatter_map_images/scatter_map_frame_{i}.png'))
    
imageio.mimsave('scatter_map_animation.gif', images, duration=0.5)  # Save as GIF
Image(filename="scatter_map_animation.gif")


# 
# # Task 3: Create a Racing Bar Plot to display the animation of EV Make and its count each year.

# In[79]:


# Approch 1

# Step 1: Aggregate data by Make and Model Year
ev_make_by_year = dataset.groupby(['Model Year', 'Make']).size().reset_index(name='EV Count')

# Step 2: Create a list of all unique makes
unique_makes = dataset['Make'].unique()

# Step 3: Ensure all makes appear in every year by filling missing combinations
all_years = pd.DataFrame({'Model Year': sorted(dataset['Model Year'].unique())})
all_combinations = all_years.assign(key=1).merge(pd.DataFrame({'Make': unique_makes, 'key':1}), on='key').drop('key', axis=1)
ev_make_by_year_full = all_combinations.merge(ev_make_by_year, on=['Model Year', 'Make'], how='left').fillna(0)

# Step 4: Convert EV Count to integer (since it was NaN before)
ev_make_by_year_full['EV Count'] = ev_make_by_year_full['EV Count'].astype(int)

# Step 5: Create the animated racing bar plot with increased height
fig = px.bar(
    ev_make_by_year_full,  # Data
    x='EV Count',  # X-axis shows the count of EVs
    y='Make',  # Y-axis shows the car Make
    color='Make',  # Color by car Make
    animation_frame='Model Year',  # Animation by year
    orientation='h',  # Horizontal bar chart
    title='Electric Vehicle Makes Over the Years',
    labels={'EV Count':'Number of EVs', 'Make':'Car Make'},  # Axis labels
    range_x=[0, ev_make_by_year_full['EV Count'].max() * 1.1],  # Dynamically set x-axis range
    height=800  # Increased height for better visibility
)

# Step 6: Show the plot
fig.show()


# In[80]:


import pandas as pd
import plotly.express as px
import imageio
import os

# Step 1: Aggregate data by Make and Model Year
ev_make_by_year = dataset.groupby(['Model Year', 'Make']).size().reset_index(name='EV Count')

# Step 2: Create a list of all unique makes
unique_makes = dataset['Make'].unique()

# Step 3: Ensure all makes appear in every year by filling missing combinations
all_years = pd.DataFrame({'Model Year': sorted(dataset['Model Year'].unique())})
all_combinations = all_years.assign(key=1).merge(pd.DataFrame({'Make': unique_makes, 'key': 1}), on='key').drop('key', axis=1)
ev_make_by_year_full = all_combinations.merge(ev_make_by_year, on=['Model Year', 'Make'], how='left').fillna(0)

# Step 4: Convert EV Count to integer
ev_make_by_year_full['EV Count'] = ev_make_by_year_full['EV Count'].astype(int)

# Step 5: Create the animated racing bar plot and save each frame as a PNG
frames = []
years = sorted(ev_make_by_year_full['Model Year'].unique())

for year in years:
    fig = px.bar(
        ev_make_by_year_full[ev_make_by_year_full['Model Year'] == year],
        x='EV Count',
        y='Make',
        color='Make',
        orientation='h',
        title=f'Electric Vehicle Makes in {year}',
        labels={'EV Count': 'Number of EVs', 'Make': 'Car Make'},
        range_x=[0, ev_make_by_year_full['EV Count'].max() * 1.1],
        height=800
    )
    
    # Save the figure as a PNG
    frame_filename = f"frame_{year}.png"
    fig.write_image(frame_filename)
    frames.append(frame_filename)

# Step 6: Create a GIF from the saved frames
with imageio.get_writer('ev_makes_racing_bar_plot.gif', mode='I', duration=0.5) as writer:
    for frame in frames:
        image = imageio.imread(frame)
        writer.append_data(image)

# Optional: Clean up the frame images
for frame in frames:
    os.remove(frame)

print("GIF created successfully as 'ev_makes_racing_bar_plot.gif'")


# In[ ]:




