import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import folium
from folium.plugins import MarkerCluster

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


