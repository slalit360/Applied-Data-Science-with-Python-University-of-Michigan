
def energy_data():
    import numpy as np
    import pandas as pd
    import re
    energy = pd.read_excel('Energy Indicators.xls', header=15, skipfooter=38, skiprows=1, na_values='...')
    energy.drop( energy.columns[[0,1]], axis=1, inplace=True)                                       
    energy.columns = ['Country','Energy Supply','Energy Supply per capita','% Renewable']
    energy['Energy Supply'] = energy['Energy Supply'].apply(lambda x : x*1000000)             
    energy['Country'] = energy['Country'].str.replace("[0-9]+","")
    energy['Country'] = energy['Country'].apply( lambda x : re.split('\(',x)[0]).str.strip()
    energy.replace({"Republic of Korea": "South Korea",
                    "United States of America": "United States",
                    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                    "China, Hong Kong Special Administrative Region": "Hong Kong"}
                   ,inplace=True)
    return energy

def gdp_data():
    import numpy as np
    import pandas as pd
    GDP = pd.read_csv('world_bank.csv', skiprows=3, header=1)
    GDP.rename(columns={'Country Name':'Country'},inplace=True)
    GDP.replace({"Korea, Rep.": "South Korea", 
                 "Iran, Islamic Rep.": "Iran",
                 "Hong Kong SAR, China": "Hong Kong"
                },inplace=True)
    return GDP

def sci_data():
    import numpy as np
    import pandas as pd
    return pd.read_excel('scimagojr-3.xlsx')

def answer_one():
    import numpy as np
    import pandas as pd
    energy = energy_data()
    GDP = gdp_data()
    ScimEn=sci_data()

    df=pd.merge(pd.merge(energy,GDP,on='Country',how='inner'), ScimEn, on='Country',how='inner')
    df.set_index('Country',inplace=True)
    a = df[(df['Rank'] < 16)]
    a = a.sort_values(by='Rank')
    col=['Rank','Documents','Citable documents','Citations','Self-citations','Citations per document','H index','Energy Supply','Energy Supply per capita','% Renewable','2006','2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    return a[col]

answer_one()

get_ipython().run_cell_magic('HTML', '', '<svg width="800" height="300">\n  <circle cx="150" cy="180" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="blue" />\n  <circle cx="200" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="red" />\n  <circle cx="100" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="green" />\n  <line x1="150" y1="125" x2="300" y2="150" stroke="black" stroke-width="2" fill="black" stroke-dasharray="5,3"/>\n  <text  x="300" y="165" font-family="Verdana" font-size="35">Everything but this!</text>\n</svg>')

def answer_two():
    import numpy as np
    import pandas as pd
    energy = energy_data()
    GDP = gdp_data()
    ScimEn=sci_data()

    df_1=pd.merge(pd.merge(energy,GDP,on='Country',how='outer'), ScimEn, on='Country',how='outer')
    df_2=pd.merge(pd.merge(energy,GDP,on='Country',how='inner'), ScimEn, on='Country',how='inner')
    lose_count=len(df_1)-len(df_2)
    return lose_count

answer_two()

def answer_three():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    years = ['2006','2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015']
    avgGDP = Top15.apply(lambda x: np.mean(x[years]), axis=1)
    return avgGDP.sort_values(ascending=False)

answer_three()

def answer_four():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    avgGDP = answer_three()
    sixth_country = avgGDP.sort_values(ascending=False).keys()[5]  #answer_three().keys()[5]
    ans = Top15.loc[sixth_country, '2015'] - Top15.loc[sixth_country, '2006']
    return ans 

answer_four()

def answer_five():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    return np.mean(Top15['Energy Supply per capita'])

answer_five()

def answer_six():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    b = np.max(Top15['% Renewable'])
    c = Top15['% Renewable'].argmax()
    return ( c, b )

answer_six()

def calc(data):
    data['citation_ratio'] = data['Self-citations'] / data['Citations']
    return data

def answer_seven():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    ratio = Top15.apply(calc,axis=1)
    max_ratio = np.max( ratio['citation_ratio'] )
    max_ratio_country = ratio['citation_ratio'].argmax()
    #max_ratio_country = ratio[ ratio['citation_ratio'] == max_ratio ].iloc[0].name
    return (max_ratio_country , max_ratio)

answer_seven()

def answer_eight():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    Top15['population'] = Top15.apply( lambda x : x['Energy Supply'] / x['Energy Supply per capita'] , axis=1)
    country=Top15.sort_values(by='population', ascending=False ).iloc[2].name
    return country

answer_eight()

def answer_nine():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    Top15['population'] = Top15.apply( lambda x : x['Energy Supply'] / x['Energy Supply per capita'] , axis=1)
    Top15['Citable_per_person'] = Top15.apply( lambda x : x['Citable documents'] / x['population'] , axis=1)
    Top15.index=Top15['Energy Supply per capita']
    correlation = Top15['Energy Supply per capita'].corr(Top15['Citable_per_person'],method='pearson') # or below one
    return correlation

answer_nine()

def plot9():
    import numpy as np
    import pandas as pd
    import matplotlib as plt
    %matplotlib inline
    Top15 = answer_one().copy()
    Top15['population'] = Top15.apply( lambda x : x['Energy Supply'] / x['Energy Supply per capita'] , axis=1)
    Top15['Citable_per_person'] = Top15.apply( lambda x : x['Citable documents'] / x['population'] , axis=1)
    return Top15.plot(x='Citable_per_person', y='Energy Supply per capita', kind='scatter', xlim=[0, 0.0006])

plot9()

def answer_ten():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    median_val = Top15['% Renewable'].median()
    Top15['HighRenew'] = Top15['% Renewable']>=median_val
    Top15['HighRenew'] = Top15['HighRenew'].apply(lambda x:1 if x else 0)
    Top15.sort_values(by='Rank', inplace=True)
    return Top15['HighRenew']

answer_ten()

def answer_eleven():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    Top15['population'] = Top15.apply( lambda x : x['Energy Supply'] / x['Energy Supply per capita'] , axis=1)
    ContinentDict  = {'China':'Asia', 
                      'United States':'North America', 
                      'Japan':'Asia', 
                      'United Kingdom':'Europe', 
                      'Russian Federation':'Europe', 
                      'Canada':'North America', 
                      'Germany':'Europe', 
                      'India':'Asia',
                      'France':'Europe', 
                      'South Korea':'Asia', 
                      'Italy':'Europe', 
                      'Spain':'Europe', 
                      'Iran':'Asia',
                      'Australia':'Australia', 
                      'Brazil':'South America'}

    Top15 = Top15.reset_index()
    Top15['Continent'] = Top15['Country'].map(ContinentDict)
    new_Top15 = Top15.groupby('Continent').agg({ 'Country': "count",
                                                'population': [sum, np.mean, np.std]
                                               })    
    new_Top15.columns=new_Top15.columns.droplevel()
    return new_Top15

answer_eleven()

def answer_twelve():
    import numpy as np
    import pandas as pd
    Top15 = answer_one().copy()
    ContinentDict  = {'China':'Asia',
                  'United States':'North America',
                  'Japan':'Asia',
                  'United Kingdom':'Europe',
                  'Russian Federation':'Europe',
                  'Canada':'North America',
                  'Germany':'Europe',
                  'India':'Asia',
                  'France':'Europe',
                  'South Korea':'Asia',
                  'Italy':'Europe',
                  'Spain':'Europe',
                  'Iran':'Asia',
                  'Australia':'Australia',
                  'Brazil':'South America'}
    Top15 = Top15.reset_index()
    Top15['Continent'] = Top15['Country'].map(ContinentDict)
    Top15['% Renewable bin'] = pd.cut(Top15['% Renewable'],5)
    new_Top15 = Top15.groupby(['Continent','% Renewable bin']).size()
    return new_Top15

answer_twelve()

def answer_thirteen():
    import numpy as np
    import pandas as pd
    import re
    Top15 = answer_one().copy()
    Top15['population'] = Top15.apply( lambda x : x['Energy Supply'] / x['Energy Supply per capita'] , axis=1)#.astype(float)
    return Top15['population'].apply(lambda x: '{0:,}'.format(x))

answer_thirteen()

def plot_optional():
    import matplotlib as plt
    %matplotlib inline
    Top15 = answer_one()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. \
This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' \
2014 GDP, and the color corresponds to the continent.")

plot_optional() 
