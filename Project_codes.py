#import the useful library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import folium
from geopy.geocoders import Nominatim
from folium.plugins import MarkerCluster
from wordcloud import WordCloud

#load the 'Crime_Data_from_2010_to_2019.csv' and 'Crime_Data_from_2020_to_Present.csv'
file_path_1 = 'Crime_Data_from_2010_to_2019.csv'
file_path_2 = 'Crime_Data_from_2020_to_Present.csv'

# Load CSV files into Pandas DataFrames
raw_data_1 = pd.read_csv(file_path_1)
raw_data_2 = pd.read_csv(file_path_2)

# Check columns and drop irrelevant ones
print(raw_data_1.columns)
print(raw_data_2.columns)

# Following are the data cleaning and some basic overview of the data.
# The steps are:
# 1) select the relevant columns
# 2) rename the columns
# 3) convert the data format
# 4) drop the error data
# 5）reset the data index
# 6) basic data overview
# 7) more detailed data visualization

#1) Select relevant columns

# We choose 19 labels from 28 labels.
data_1 = raw_data_1[['DATE OCC', 'TIME OCC', 'AREA ', 'AREA NAME', 'Crm Cd', 'Crm Cd Desc', 'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'Premis Desc', 'Weapon Used Cd', 'Weapon Desc', 'Crm Cd 1',
       'Crm Cd 2', 'LOCATION', 'Cross Street', 'LAT', 'LON']]
data_2 = raw_data_2[['DATE OCC', 'TIME OCC', 'AREA', 'AREA NAME', 'Crm Cd', 'Crm Cd Desc', 'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'Premis Desc', 'Weapon Used Cd', 'Weapon Desc', 'Crm Cd 1',
       'Crm Cd 2', 'LOCATION', 'Cross Street', 'LAT', 'LON']]

# 2) rename the columns
# Rename the 'AREA ' for data_1 to delete the space
data_1.rename(columns={'AREA ': 'AREA'}, inplace = True)
# Your original columns
columns = ['DATE OCC', 'TIME OCC', 'AREA', 'AREA NAME', 'Crm Cd', 'Crm Cd Desc', 'Vict Age', 'Vict Sex', 'Vict Descent', 'Premis Cd', 'Premis Desc', 'Weapon Used Cd', 'Weapon Desc', 'Crm Cd 1', 'Crm Cd 2', 'LOCATION', 'Cross Street', 'LAT', 'LON']

# Convert columns to uppercase
columns_upper = [col.upper().strip() for col in columns]

# Rename columns in the DataFrame
data_1.columns = columns_upper
data_2.columns = columns_upper
# Now, all column titles are in uppercase
# print(data_1.head())  # Print the first few rows to verify the changes


# 3) convert the data format

# Convert 'DATE OCC' to datetime format
data_1['DATE OCC'] = pd.to_datetime(data_1['DATE OCC'])

# Extract Year, Month, and Day into separate columns
data_1['YEAR'] = data_1['DATE OCC'].dt.year
data_1['MONTH'] = data_1['DATE OCC'].dt.month
data_1['DAY'] = data_1['DATE OCC'].dt.day

# Print the first few rows to verify changes
print(data_1[['DATE OCC', 'YEAR', 'MONTH', 'DAY']].head())

# Convert 'TIME OCC' to string and then to time format
data_1['TIME OCC'] = pd.to_datetime(data_1['TIME OCC'], format='%H%M', errors='coerce').dt.strftime('%H:%M')

# Print the first few rows to verify changes
print(data_1['TIME OCC'].head())


# Convert 'DATE OCC' to datetime format
data_2['DATE OCC'] = pd.to_datetime(data_2['DATE OCC'])

# Extract Year, Month, and Day into separate columns
data_2['YEAR'] = data_2['DATE OCC'].dt.year
data_2['MONTH'] = data_2['DATE OCC'].dt.month
data_2['DAY'] = data_2['DATE OCC'].dt.day

# Print the first few rows to verify changes
print(data_2[['DATE OCC', 'YEAR', 'MONTH', 'DAY']].head())

# Convert 'TIME OCC' to string and then to time format
data_2['TIME OCC'] = pd.to_datetime(data_2['TIME OCC'], format='%H%M', errors='coerce').dt.strftime('%H:%M')

# Print the first few rows to verify changes
print(data_2['TIME OCC'].head())

# 4) drop the error data

# Filter out rows where victim age is between -12 and 0
data_1 = data_1[(data_1['VICT AGE'] > 0) | (data_1['VICT AGE'].isnull())]

# Check the new victim age range after filtering
print("Updated Victim Age Range:")
print("Minimum Age:", data_1['VICT AGE'].min())
print("Maximum Age:", data_1['VICT AGE'].max())

# Filter out rows where victim age is between -12 and 0
data_2 = data_2[(data_2['VICT AGE'] > 0) | (data_2['VICT AGE'].isnull())]

# Check the new victim age range after filtering
print("Updated Victim Age Range:")
print("Minimum Age:", data_2['VICT AGE'].min())
print("Maximum Age:", data_2['VICT AGE'].max())

# There are some 2021 data in the data_1, need to drop
# Filter out rows where victim age is between -12 and 0
data_1 = data_1[(data_1['YEAR'] < 2020) | (data_1['YEAR'].isnull())]


# 5）reset the data index

# Reset the data Index
data_1 = data_1.reset_index(drop=True)

data_2 = data_2.reset_index(drop=True)

# Displaying the DataFrame to check the new order
# print(data_1)

# Print all column names of data_1
print("Column names of data_1:")
print(data_1.columns.tolist())

# Print all column names of data_2
print("Column names of data_2:")
print(data_2.columns.tolist())

'''
Here are some descritions of the data columns:

Data_1 is the pd.Frame of data before 2020.
Data_2 is the pd.Frame of data after 2020.

'YEAR' :                year of the crime happened. (e.g. 2019)
'MONTH' :               Month of the crime happened. (e.g. 3)
'DAY' :                 Day of the crime happened. (e.g. 11)
'TIME OCC' :            Exact time of the crime happened. (e.g. 10:13)

'AREA':                 The LAPD has 21 Community Police Stations referred to as Geographic Areas within the department. 
                        These Geographic Areas are sequentially numbered from 1-21.
'AREA NAME':            ['Newton' 'Hollywood' 'Central' 'Southwest' 'Devonshire' 'Rampart'
                        'Olympic' 'Northeast' 'Harbor' '77th Street' 'Hollenbeck' 'Pacific'
                        'Wilshire' 'West LA' 'Southeast' 'Topanga' 'Mission' 'West Valley'
                        'Van Nuys' 'N Hollywood' 'Foothill']

'CRM CD':               Indicates the crime committed. (Same as Crime Code 1).
'CRM CD DESC':          Defines the Crime Code provided.
'CRM CD 1':             Indicates the crime committed. Crime Code 1 is the primary and most serious one. 
                        Crime Code 2, 3, and 4 are respectively less serious offenses. Lower crime class numbers are more serious.
'CRM CD 2':             May contain a code for an additional crime, less serious than Crime Code 1.

'VICT AGE':             The age of the victim. 
'VICT SEX':             The sex of the victim.
'VICT DESCENT':         Descent Code: A - Other Asian B - Black C - Chinese D - Cambodian F - Filipino 
                        G - Guamanian H - Hispanic/Latin/Mexican I - American Indian/Alaskan Native 
                        J - Japanese K - Korean L - Laotian O - Other P - Pacific Islander S - Samoan 
                        U - Hawaiian V - Vietnamese W - White X - Unknown Z - Asian Indian

'PREMIS CD':            The type of structure, vehicle, or location where the crime took place.
'PREMIS DESC':          Defines the Premise Code provided.

'WEAPON USED CD':       The type of weapon used in the crime.
'WEAPON DESC':          Defines the Weapon Used Code provided.


'LOCATION':             Street address of crime incident rounded to the nearest hundred block to maintain anonymity.
'CROSS STREET':         Cross Street of rounded Address.
'LAT':                  Latitude
'LON':                  Longtitude


'''

# 6) basic data overview - part 1

# Output some basic information of the data

# For Data_1
print("For the crime data before the covid 19:")
print()

# How many different areas?
print("There are " + str(data_1['AREA NAME'].nunique()) + " areas")
print("The Area Name are:", data_1['AREA NAME'].unique())
print()

# How many different crime types?
print("There are " + str(data_1['CRM CD DESC'].nunique()) + " types of the crime")
print()

# What is the Vict Range
print("Victim Age Range:")
print("Minimum Age:", data_1['VICT AGE'].min())
print("Maximum Age:", data_1['VICT AGE'].max())
print()

# How many different premis types?
num_premise_types = data_1['PREMIS DESC'].nunique()
print("There are " + str(num_premise_types) + " different premise types")
print()

# How many different weapons used?
num_weapon_types = data_1['WEAPON DESC'].nunique()
print("There are " + str(num_weapon_types) + " different weapons used")
print()

# How many different Crm Cd 1 types?
num_crm_cd_1_types = data_1['CRM CD 1'].nunique()
print("There are " + str(num_crm_cd_1_types) + " different Crm Cd 1 types")

# 6) basic data overview - part 2
# For Data_2
print("For the crime data after the covid 19:")
print()

# How many different areas?
print("There are " + str(data_2['AREA NAME'].nunique()) + " areas")
print("The Area Name are:", data_2['AREA NAME'].unique())
print()

# How many different crime types?
print("There are " + str(data_2['CRM CD DESC'].nunique()) + " types of the crime")
print()

# What is the Vict Range
print("Victim Age Range:")
print("Minimum Age:", data_2['VICT AGE'].min())
print("Maximum Age:", data_2['VICT AGE'].max())
print()

# How many different premis types?
num_premise_types = data_2['PREMIS DESC'].nunique()
print("There are " + str(num_premise_types) + " different premise types")
print()

# How many different weapons used?
num_weapon_types = data_2['WEAPON DESC'].nunique()
print("There are " + str(num_weapon_types) + " different weapons used")
print()

# How many different Crm Cd 1 types?
num_crm_cd_2_types = data_2['CRM CD 1'].nunique()
print("There are " + str(num_crm_cd_2_types) + " different Crm Cd 1 types")

# 7) more detailed data visualization - part 1

# a. the crime event number tendency for year before and after covid:
print(data_1['YEAR'].unique())
# 'Year' is the column representing the year in data_1
crime_count_by_year = data_1['YEAR'].value_counts().sort_index()

# Plotting the crime count tendency by year
plt.figure(figsize=(10, 8))
crime_count_by_year.plot(kind='line', marker='o', color='blue')
plt.title('Crime Number Tendency by Year (before covid)')
plt.xlabel('YEAR')
plt.ylabel('Crime Count')
plt.grid(True)
plt.show()

# 'Year' is the column representing the year in data_2
print(data_2['YEAR'].unique())
crime_count_by_year = data_2['YEAR'].value_counts().sort_index()

# Plotting the crime count tendency by year
plt.figure(figsize=(10, 8))
crime_count_by_year.plot(kind='line', marker='o', color='blue')
plt.title('Crime Number Tendency by Year (after covid)')
plt.xlabel('YEAR')
plt.ylabel('Crime Count')
plt.grid(True)
plt.show()


# 7) more detailed data visualization - part 2
'''
'Vict Age':             The age of the victim. 
'Vict Sex':             The sex of the victim.
'Vict Descent':         Descent Code: A - Other Asian B - Black C - Chinese D - Cambodian F - Filipino 
                        G - Guamanian H - Hispanic/Latin/Mexican I - American Indian/Alaskan Native 
                        J - Japanese K - Korean L - Laotian O - Other P - Pacific Islander S - Samoan 
                        U - Hawaiian V - Vietnamese W - White X - Unknown Z - Asian Indian
'''
print("For the crime data before covid:")
# Victim's Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data_1['VICT AGE'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Victim Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Victim's Sex distribution
plt.figure(figsize=(6, 6))
data_1['VICT SEX'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Victim Sex Distribution')
plt.ylabel('')
plt.show()

# Victim's Descent distribution
plt.figure(figsize=(10, 6))
descent_counts = data_1['VICT DESCENT'].value_counts()
sns.barplot(x=descent_counts.index, y=descent_counts.values, palette='viridis')
plt.title('Victim Descent Distribution')
plt.xlabel('Descent Code')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 7) more detailed data visualization - part 2
'''
'Vict Age':             The age of the victim. 
'Vict Sex':             The sex of the victim.
'Vict Descent':         Descent Code: A - Other Asian B - Black C - Chinese D - Cambodian F - Filipino 
                        G - Guamanian H - Hispanic/Latin/Mexican I - American Indian/Alaskan Native 
                        J - Japanese K - Korean L - Laotian O - Other P - Pacific Islander S - Samoan 
                        U - Hawaiian V - Vietnamese W - White X - Unknown Z - Asian Indian
'''

print("For the crime data after covid:")
# Victim's Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(data_2['VICT AGE'].dropna(), bins=30, kde=True, color='skyblue')
plt.title('Distribution of Victim Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Victim's Sex distribution
plt.figure(figsize=(6, 6))
data_2['VICT SEX'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title('Victim Sex Distribution')
plt.ylabel('')
plt.show()

# Victim's Descent distribution
plt.figure(figsize=(10, 6))
descent_counts = data_2['VICT DESCENT'].value_counts()
sns.barplot(x=descent_counts.index, y=descent_counts.values, palette='viridis')
plt.title('Victim Descent Distribution')
plt.xlabel('Descent Code')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 7) more detailed data visualization - part 3
'''
'Premis Cd':            The type of structure, vehicle, or location where the crime took place.
'Premis Desc':          Defines the Premise Code provided.

'Weapon Used Cd':       The type of weapon used in the crime.
'Weapon Desc':          Defines the Weapon Used Code provided.

'''
print("For the crime data before covid:")

# Plotting Premise Description distribution
plt.figure(figsize=(10, 6))
premis_counts = data_1['PREMIS DESC'].value_counts()
shortened_labels = [label[:15] + '...' if len(label) > 15 else label for label in premis_counts.index[:10]]
sns.barplot(x=shortened_labels, y=premis_counts.values[:10], palette='coolwarm')
plt.title('Top 10 Premise Descriptions')
plt.xlabel('Premise Description')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting Weapon Description distribution
plt.figure(figsize=(10, 6))
weapon_counts = data_1['WEAPON DESC'].value_counts()
shortened_labels = [label[:15] + '...' if len(label) > 15 else label for label in weapon_counts.index[:10]]
sns.barplot(x=shortened_labels, y=weapon_counts.values[:10], palette='muted')
plt.title('Top 10 Weapon Descriptions')
plt.xlabel('Weapon Description')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7) more detailed data visualization - part 3
'''
'Premis Cd':            The type of structure, vehicle, or location where the crime took place.
'Premis Desc':          Defines the Premise Code provided.

'Weapon Used Cd':       The type of weapon used in the crime.
'Weapon Desc':          Defines the Weapon Used Code provided.

'''
print("For the crime data after covid:")

# Plotting Premise Description distribution
plt.figure(figsize=(10, 6))
premis_counts = data_2['PREMIS DESC'].value_counts()
shortened_labels = [label[:15] + '...' if len(label) > 15 else label for label in premis_counts.index[:10]]
sns.barplot(x=shortened_labels, y=premis_counts.values[:10], palette='coolwarm')
plt.title('Top 10 Premise Descriptions')
plt.xlabel('Premise Description')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting Weapon Description distribution
plt.figure(figsize=(10, 6))
weapon_counts = data_2['WEAPON DESC'].value_counts()
shortened_labels = [label[:15] + '...' if len(label) > 15 else label for label in weapon_counts.index[:10]]
sns.barplot(x=shortened_labels, y=weapon_counts.values[:10], palette='muted')
plt.title('Top 10 Weapon Descriptions')
plt.xlabel('Weapon Description')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 7) more detailed data visualization - part 4
print("For the crime data before covid:")
# Filter data for criminal homicides and remove entries with latitude or longitude as 0
homicides = data_1[data_1['CRM CD DESC'] == 'CRIMINAL HOMICIDE']
homicides = homicides[(homicides['LAT'] != 0) & (homicides['LON'] != 0)]

# Create a base map centered at a specific location (Los Angeles coordinates as an example)
m = folium.Map(location=[34.052235, -118.243683], zoom_start=12)

# Add markers for each homicide location
marker_cluster = MarkerCluster().add_to(m)

for index, row in homicides.iterrows():
    folium.Marker(
        location=[row['LAT'], row['LON']],
        popup=row['CRM CD DESC'],
        icon=None  # You can customize the icon here if needed
    ).add_to(marker_cluster)

# Display the map

# 7) more detailed data visualization - part 4
print("For the crime data before covid:")
# Filter data for criminal homicides and remove entries with latitude or longitude as 0
homicides = data_2[data_2['CRM CD DESC'] == 'CRIMINAL HOMICIDE']
homicides = homicides[(homicides['LAT'] != 0) & (homicides['LON'] != 0)]

# Create a base map centered at a specific location (Los Angeles coordinates as an example)
m = folium.Map(location=[34.052235, -118.243683], zoom_start=12)

# Add markers for each homicide location
marker_cluster = MarkerCluster().add_to(m)

for index, row in homicides.iterrows():
    folium.Marker(
        location=[row['LAT'], row['LON']],
        popup=row['CRM CD DESC'],
        icon=None  # You can customize the icon here if needed
    ).add_to(marker_cluster)

# Display the map


#---------------------------Gayatri Question--------------------------------#

# 1) Part One.
data_2017_2019 = data_1[(data_1['YEAR'] >= 2017) & (data_1['YEAR'] <= 2019)]
# Filter data for the years 2020 to 2023 from data_2
data_2020_2023 = data_2[(data_2['YEAR'] >= 2020) & (data_2['YEAR'] <= 2023)]

# Group by location to get the total number of crimes for each neighborhood
crime_by_location_2010_2019 = data_2017_2019.groupby('AREA NAME').size().reset_index(name='Crime_Count_2010_2019')
crime_by_location_2020_present = data_2020_2023.groupby('AREA NAME').size().reset_index(name='Crime_Count_2020_Present')

# Merge the two dataframes on the 'AREA NAME' column
merged_data = pd.merge(crime_by_location_2010_2019, crime_by_location_2020_present, on='AREA NAME', how='outer').fillna(0)


# Calculate the percentage change in crime counts
merged_data['Percentage_Change'] = ((merged_data['Crime_Count_2020_Present'] - merged_data['Crime_Count_2010_2019']) / merged_data['Crime_Count_2010_2019']) * 100

# Sort the data by percentage change to identify areas most affected
sorted_data = merged_data.sort_values(by='Percentage_Change', ascending=False)

# Plot the affected areas
top_n=21
plt.figure(figsize=(12, 8))
sns.barplot(x='Percentage_Change', y='AREA NAME', data=sorted_data.head(top_n), palette='viridis')
plt.title('Areas Affected by Crime Rate Change'.format(top_n))
plt.xlabel('Percentage Change in Crime Rate')
plt.ylabel('AREA NAME')
plt.show()


# 2) Part Two.
data_2017_2019 = data_1[(data_1['YEAR'] >= 2017) & (data_1['YEAR'] <= 2019)]
# Filter data for the years 2020 to 2023 from data_2
data_2020_2023 = data_2[(data_2['YEAR'] >= 2020) & (data_2['YEAR'] <= 2023)]

# Group by location to get the total number of crimes for each neighborhood
crime_by_location_2010_2019 = data_2017_2019.groupby('AREA NAME').size().reset_index(name='Crime_Count_2010_2019')
crime_by_location_2020_present = data_2020_2023.groupby('AREA NAME').size().reset_index(name='Crime_Count_2020_Present')

# Merge the two dataframes on the 'AREA NAME' column
merged_data_location = pd.merge(crime_by_location_2010_2019, crime_by_location_2020_present, on='AREA NAME', how='outer').fillna(0)

# Geocode addresses to obtain latitude and longitude
geolocator = Nominatim(user_agent="crime_analysis")

# Define the latitude and longitude boundaries of Los Angeles
la_lat_min, la_lat_max = 33.5, 34.9
la_lon_min, la_lon_max = -118.7, -118.1

#lambda function to geocode each 'AREA NAME' within Los Angeles
merged_data_location['location'] = merged_data_location['AREA NAME'].apply(lambda x: geolocator.geocode(f"{x}, Los Angeles") if x else None)

# Drop rows with missing 'location' values
merged_data_location = merged_data_location.dropna(subset=['location'])

# Filter out locations that are not in the specified boundaries of Los Angeles
merged_data_location = merged_data_location[(merged_data_location['location'].apply(lambda loc: la_lat_min <= loc.latitude <= la_lat_max) & merged_data_location['location'].apply(lambda loc: la_lon_min <= loc.longitude <= la_lon_max))]

# Extract latitude and longitude
merged_data_location['LAT'] = merged_data_location['location'].apply(lambda loc: loc.latitude if loc else None)
merged_data_location['LON'] = merged_data_location['location'].apply(lambda loc: loc.longitude if loc else None)

# Drop the 'location' column if you no longer need it
merged_data_location = merged_data_location.drop(columns=['location'])

# Create a Folium map
m = folium.Map(location=[34.0522, -118.2437], zoom_start=10)

# Add markers to the map
merged_data_location['Percentage_Change'] = ((merged_data_location['Crime_Count_2020_Present'] - merged_data_location['Crime_Count_2010_2019']) / merged_data_location['Crime_Count_2010_2019']) * 100
for idx, row in merged_data_location.iterrows():
    popup_text = f"Location: {row['AREA NAME']}<br>Percentage Change: {row['Percentage_Change']:.2f}%"
    folium.Marker(location=[row['LAT'], row['LON']], popup=popup_text).add_to(m)

# Display the map

#3) Part Three.
data_2017_2019 = data_1[(data_1['YEAR'] >= 2017) & (data_1['YEAR'] <= 2019)]

# Define the dimensions for the plot
first_dimension = "AREA NAME"
second_dimension = "VICT SEX"

# Specify the desired area names
area_names = ['Central','Olympic','Hollywood','Wilshire','Pacific','Newton','Rampart','Southeast','Hollenbeck','West LA']

# Filter the DataFrame for specific area names
filtered_data1 = data_2017_2019[data_2017_2019[first_dimension].isin(area_names)]

# Set up the figure and axis
plt.figure(figsize=(20, 15))

# Use Seaborn's countplot to visualize the data
sns.countplot(
    y=first_dimension,           # Variable for the vertical axis
    hue=second_dimension,        # Variable for the color categories
    data=filtered_data1,          # DataFrame containing the data
    orient="h",                     # Set the orientation to horizontal
    palette="viridis"            # Choose a color palette
)

# Set plot title
plt.title(f"{first_dimension} - {second_dimension}, Before COVID")

# Display the plot
plt.show()


# 4) Part Four

# Filter data for the years 2020 to 2023 from data_2
data_2020_2023 = data_2[(data_2['YEAR'] >= 2020) & (data_2['YEAR'] <= 2023)]

# Define the dimensions for the plot
first_dimension = "AREA NAME"
second_dimension = "VICT SEX"

# Specify the desired area names
area_names = ['Central','Olympic','Hollywood','Wilshire','Pacific','Newton','Rampart','Southeast','Hollenbeck','West LA']

# Filter the DataFrame for specific area names
filtered_data2 = data_2020_2023[data_2020_2023[first_dimension].isin(area_names)]

# Set up the figure and axis
plt.figure(figsize=(20, 15))

# Use Seaborn's countplot to visualize the data
sns.countplot(
    y=first_dimension,           # Variable for the vertical axis
    hue=second_dimension,        # Variable for the color categories
    data=filtered_data2,          # DataFrame containing the data
    orient="h",                  # Set the orientation to horizontal
    palette="viridis"            # Choose a color palette
)

# Set plot title
plt.title(f"{first_dimension} - {second_dimension}, After COVID")

# Display the plot
plt.show()


#---------------------------------Vaishnavi_code--------------------------------#
def plot_word_plots_data_1(column_name):
  '''This function plots word plots for given dataframe columns
  param- text: converts values in colums to a string
  '''
  assert isinstance(column_name,str)
  text = " ".join(cat.split()[0] for cat in data_1[column_name])
  word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

  plt.imshow(word_cloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()


def plot_gender_ratio_data_1(crime):
  "This function plots the gender ratio for a particular crime for data_1"
  gk=data_1.groupby(['CRM CD DESC'])
  jk=gk.get_group(crime).groupby(['VICT SEX']).count()
  fig = plt.figure()
  langs = ['Female','Male']
  students = [jk['DATE OCC'][0], jk['DATE OCC'][1]]
  ax = fig.add_axes([0,0,0.2,0.5])
  ax.bar(langs,students)
  plt.show()

def return_avg_age_data_1(crime):
  '''This function returns the average age of the victims'''
  gk=data_1.groupby(['CRM CD DESC'])
  jk=int(gk.get_group(crime)['VICT AGE'].mean())



def plot_weapons_use_data_1(crime):
  '''This function plots the weapons used in the crime'''
  fk=data_1.groupby(['CRM CD DESC']).get_group(crime)
  sk=fk['WEAPON DESC']
  unique_values, counts = np.unique(sk, return_counts=True)
  plt.figure(figsize=(20, 3))
  plt.xticks(rotation=25)
  plt.bar(unique_values, counts, width=0.4)


def plot_frequency_in_area_data_1(crime):
  '''This function plots the number of times a crime has occured in the particular area'''
  fk=data_1.groupby(['CRM CD DESC']).get_group(crime)
  sk=fk['AREA NAME']
  unique_values, counts = np.unique(sk, return_counts=True)
  plt.figure(figsize=(3, 3))
  plt.xticks(rotation=25)
  plt.bar(unique_values, counts, width=0.4)
  
  
def plot_word_plots_data_2(column_name):
  '''This function plots word plots for given dataframe columns
  param- text: converts values in colums to a string
  '''
  assert isinstance(column_name,str)
  text = " ".join(cat.split()[0] for cat in data_2[column_name])
  word_cloud = WordCloud(collocations = False, background_color = 'white').generate(text)

  plt.imshow(word_cloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()


def plot_gender_ratio_data_2(crime):
  "This function plots the gender ratio for a particular crime for data_2"
  gk=data_2.groupby(['CRM CD DESC'])
  jk=gk.get_group(crime).groupby(['VICT SEX']).count()
  fig = plt.figure()
  langs = ['Female','Male']
  students = [jk['DATE OCC'][0], jk['DATE OCC'][1]]
  ax = fig.add_axes([0,0,0.2,0.5])
  ax.bar(langs,students)
  plt.show()

def return_avg_age_data_2(crime):
  '''This function returns the average age of the victims'''
  gk=data_2.groupby(['CRM CD DESC'])
  jk=int(gk.get_group(crime)['VICT AGE'].mean())



def plot_weapons_use_data_2(crime):
  '''This function plots the weapons used in the crime'''
  fk=data_2.groupby(['CRM CD DESC']).get_group(crime)
  sk=fk['WEAPON DESC']
  unique_values, counts = np.unique(sk, return_counts=True)
  plt.figure(figsize=(20, 3))
  plt.xticks(rotation=25)
  plt.bar(unique_values, counts, width=0.4)


def plot_frequency_in_area_data_2(crime):
  '''This function plots the number of times a crime has occured in the particular area'''
  fk=data_2.groupby(['CRM CD DESC']).get_group(crime)
  sk=fk['AREA NAME']
  unique_values, counts = np.unique(sk, return_counts=True)
  plt.figure(figsize=(3, 3))
  plt.xticks(rotation=25)
  plt.bar(unique_values, counts, width=0.4)
  
plot_word_plots_data_1('AREA NAME')
plot_word_plots_data_2('AREA NAME')
plot_frequency_in_area_data_1('CHILD STEALING')
plot_frequency_in_area_data_2('CHILD STEALING')



#---------------------------------------------------Darren Code------------------------------------------------------------#
def plot_crime_instances(db, crime_amt =10, label_len = 30, perc = False):
    '''
    Description: Take in the crime database and count the values of each type of crime and order from greatest to least greatest and then plot them in a bar graph
    crime_amt determines the top n crimes to plot
    label_len determines the length of the labels on the x axis

    input: db - Pandas DataFrame - crime database from LA county
    crime_amt - int - number of crimes to plot
    label_len - int - length of labels
    perc - Boolean - If true output the y axis as percentages, else output the y axis as the instances per year

    output: returns a pandas Series counting the amount of instances of each crime and plots them

    EX: 
    perc = True
    returns 
                                                               Crime Description
    BATTERY - SIMPLE ASSAULT                                   1.079829e-01
    BURGLARY FROM VEHICLE                                      9.156555e-02
    THEFT PLAIN - PETTY ($950 & UNDER)                         7.501745e-02
    BURGLARY                                                   7.273924e-02
    THEFT OF IDENTITY                                          7.195941e-02
                                                               ...     
    TRAIN WRECKING                                             1.167414e-06
    BLOCKING DOOR INDUCTION CENTER                             1.167414e-06
    FIREARMS RESTRAINING ORDER (FIREARMS RO)                   1.167414e-06
    DRUNK ROLL - ATTEMPT                                       5.837071e-07
    FIREARMS TEMPORARY RESTRAINING ORDER (TEMP FIREARMS RO)    5.837071e-07


    perc = False
    returns 
                                                               Crime Description
    BATTERY - SIMPLE ASSAULT                                   18499.5
    BURGLARY FROM VEHICLE                                      15686.9
    THEFT PLAIN - PETTY ($950 & UNDER)                         12851.9
    BURGLARY                                                   12461.6
    THEFT OF IDENTITY                                          12328.0
                                                            ...   
    TRAIN WRECKING                                                 0.2
    BLOCKING DOOR INDUCTION CENTER                                 0.2
    FIREARMS RESTRAINING ORDER (FIREARMS RO)                       0.2
    DRUNK ROLL - ATTEMPT                                           0.1
    FIREARMS TEMPORARY RESTRAINING ORDER (TEMP FIREARMS RO)        0.1

    '''

    assert isinstance(db,pd.DataFrame), "db is not a Pandas DataFrame!\n"
    assert 'CRM CD DESC' in db, "The database does not include Crime Descriptions!\n"
    assert isinstance(crime_amt,int), "The crime amount is not an Int!\n"
    assert crime_amt > 0 , "The crime amount is not a valid value!\n"
    assert isinstance(label_len,int), "The Label Length is not an Int!\n"
    assert label_len > 0 , "The Label Length is not a valid value!\n"
    assert isinstance(perc,bool), "perc is not a boolean value!\n"

    crime_type = db['CRM CD DESC'].value_counts()
    if perc:#turn y axis into percentages
        tot_crime = crime_type.sum()
        crime_type = crime_type/tot_crime
    else:# turn y axis into instances per year
        assert 'YEAR' in db, "The database does not include the years!"
        amt_years = db['YEAR'].max() - db['YEAR'].min()+1
        crime_type = crime_type/amt_years


    #shorten the labels to fit well
    shortened_labels = [label[:label_len] + '...' if len(label) > label_len else label for label in crime_type.index[:crime_amt]]
    plt.figure(figsize = (12,10))
    sns.barplot(x=shortened_labels, y=crime_type[:crime_amt], palette='colorblind')
    plt.title('Top 10 Crime Types')
    plt.xlabel('Crime Description')
    if perc:
        plt.ylabel('Percentage (%)')
    else:
        plt.ylabel('Count/year')
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return crime_type

def plot_by_ages(db, crime_amt = 10, label_len = 30, perc = False):
    '''
    group the crimes by the age groups of the victims (0,25], (25,50], (50,75] and plot a bar graph with the top n crimes

    returns a list of pandas Series
    '''
    assert isinstance(db,pd.DataFrame), "db is not a Pandas DataFrame!\n"
    assert 'CRM CD DESC' in db, "The database does not include Crime Descriptions!\n"
    assert 'VICT AGE' in db, "The database does not include the victims ages!\n"
    assert isinstance(crime_amt,int), "The crime amount is not an Int!\n"
    assert crime_amt > 0 , "The crime amount is not a valid value!\n"
    assert isinstance(label_len,int), "The Label Length is not an Int!\n"
    assert label_len > 0 , "The Label Length is not a valid value!\n"
    assert isinstance(perc,bool), "perc is not a boolean value!\n"

    ages = list(range(0,76,25))
    age_db = db.groupby(pd.cut(db['VICT AGE'], ages)).indices
    age_list = []
    for age in age_db:
        plt.figure(figsize=(12, 10))
        age_crime_type = db.iloc[age_db[age]]['CRM CD DESC'].value_counts()
        if perc:#turn y axis into percentages
            tot_crime = age_crime_type.sum()
            age_crime_type = age_crime_type/tot_crime
        else:# turn y axis into instances per year
            assert 'YEAR' in db, "The database does not include the years!"
            amt_years = db['YEAR'].max() - db['YEAR'].min()+1
            age_crime_type = age_crime_type/amt_years
        age_list.append(age_crime_type)

        shortened_labels = [label[:label_len] + '...' if len(label) > label_len else label for label in age_crime_type.index[:crime_amt]]
        age_tot_crime = age_crime_type.sum()
        sns.barplot(x=shortened_labels, y=age_crime_type[:crime_amt], palette='colorblind')
        plt.title('Top 10 Crime Types Against Ages : ' + str(age))
        plt.xlabel('Crime Description')
        if perc:
            plt.ylabel('Percentage (%)')
        else:
            plt.ylabel('Count/year')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return age_list


def plot_by_sex(db, crime_amt = 10, label_len = 30, perc = False):
    '''
    group the crimes by sexes of the victims M or F and plot a bar graph with the top n crimes

    returns a list of pandas Series
    '''
    assert isinstance(db,pd.DataFrame), "db is not a Pandas DataFrame!\n"
    assert 'CRM CD DESC' in db, "The database does not include Crime Descriptions!\n"
    assert 'VICT SEX' in db, "The database does not include the victims sex!\n"
    assert isinstance(crime_amt,int), "The crime amount is not an Int!\n"
    assert crime_amt > 0 , "The crime amount is not a valid value!\n"
    assert isinstance(label_len,int), "The Label Length is not an Int!\n"
    assert label_len > 0 , "The Label Length is not a valid value!\n"
    assert isinstance(perc,bool), "perc is not a boolean value!\n"
    sex = db['VICT SEX'].value_counts().index[:2]
    sex_list = []
    for s in sex:
        sex_db = db.groupby(db['VICT SEX']).get_group(s)
        plt.figure(figsize=(12, 10))
        sex_crime_type = sex_db['CRM CD DESC'].value_counts()
        if perc:#turn y axis into percentages
            tot_crime = sex_crime_type.sum()
            sex_crime_type = sex_crime_type/tot_crime
        else:# turn y axis into instances per year
            assert 'YEAR' in db, "The database does not include the years!"
            amt_years = db['YEAR'].max() - db['YEAR'].min()+1
            sex_crime_type = sex_crime_type/amt_years
        sex_list.append(sex_crime_type)
        shortened_labels = [label[:label_len] + '...' if len(label) > label_len else label for label in sex_crime_type.index[:crime_amt]]
        sex_tot_crime = sex_crime_type.sum()
        sns.barplot(x=shortened_labels, y=sex_crime_type[:crime_amt], palette='colorblind')
        plt.title('Top 10 Crime Types Against Sex : ' + s)
        plt.xlabel('Crime Description')
        if perc:
            plt.ylabel('Percentage (%)')
        else:
            plt.ylabel('Count/year')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return sex_list

def plot_by_asian(db):
    '''
    Group crimes by asian descent vs non asian descent and plot a pie chart with crimes commited against asians vs non asians

    returns a list of pandas Series, [asian_crime, non_asian_crime]

    '''
    assert isinstance(db,pd.DataFrame), "db is not a Pandas DataFrame!\n"
    assert 'CRM CD DESC' in db, "The database does not include Crime Descriptions!\n"
    assert 'VICT DESCENT' in db, "The database does not include the victims descent!\n"


    '''
    'VICT DESCENT':         Descent Code: A - Other Asian B - Black C - Chinese D - Cambodian F - Filipino 
                            G - Guamanian H - Hispanic/Latin/Mexican I - American Indian/Alaskan Native 
                            J - Japanese K - Korean L - Laotian O - Other P - Pacific Islander S - Samoan 
                            U - Hawaiian V - Vietnamese W - White X - Unknown Z - Asian Indian
    '''

    asian = ['A', 'C', 'D','F', 'J', 'K', 'L', 'V', 'Z']
    non_asian = ['B', 'G', 'H', 'I', 'O', 'P', 'S','U', 'W', 'X']
    labels = ['Asian Descent', 'Non Asian Descent']
    asian_crime = db['VICT DESCENT'].value_counts()[asian].sum()
    non_asian_crime = db['VICT DESCENT'].value_counts()[non_asian].sum()
    plt.figure()
    plt.pie([asian_crime,non_asian_crime], labels = labels,autopct='%1.1f%%')
    plt.title("Crimes commited against people of asian vs non asian descent")
    return [asian_crime, non_asian_crime]
