#!/usr/bin/env python
# coding: utf-8

# In[2]:


import re
import time
import sqlite3
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
# Get the webpage 
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--disable-extensions")
browser = webdriver.Chrome(executable_path=ChromeDriverManager().install(), chrome_options=chrome_options)
browser.get("https://www.tripadvisor.com/AttractionProductReview-g60745-d11450485-Boston_Hop_On_Hop_Off_Trolley_Tour_with_14_Stops-Boston_Massachusetts.html")
ULR_pattern_str = "https://www.tripadvisor.com/AttractionProductReview-g60745-d11450485-Boston_Hop_On_Hop_Off_Trolley_Tour_with_14_Stops-Boston_Massachusetts.html"
page_URL= ULR_pattern_str.replace("$NUM", str(1))
browser.get(ULR_pattern_str)
page_content = browser.page_source
#Parsing the first page to calculate the maximal number of review pages
num_of_pages = 0;
g = re.compile(r"KxBGd\">(\d,\d*?) reviews",re.S|re.I).findall(page_content)  
if (len(g) > 0):
    total_num_reviews = g[0].strip().replace(',', '')
    print("Total Number of Reviews: " + str(total_num_reviews))
    num_of_pages = int(total_num_reviews) // 10 + 1
    #prepare the database for storing results:
conn = sqlite3.connect('group.db')
c = conn.cursor()
c.execute('''DROP TABLE Tripadvisorreviewstable ''')
c.execute("CREATE TABLE Tripadvisorreviewstable(            name varchar(100),             rating varchar(100),            contribution varchar(50),            userLocation varchar(50),            reviewMessage text(1000),            month varchar(50),             year varchar(50))") 
#All listed product information
n = 10
for i in range (num_of_pages):
    all_chunks=re.compile(r'reviewCard(.*?)biGQs _P pZUbB mowmC',re.S|re.I).findall(page_content) 
    if len(all_chunks)>0:
        for chunk in all_chunks:
            #print(chunk)
            #Initialization
            name=""
            rating=""
            contribution=""
            user_location=""
            month=""
            year=""
        
            #Parsing name
            matches=re.compile(r'<a target="_self" href=".*?" class="BMQDV _F G- wSSLS SwZTJ FGwzt ukgoS">(.*?)<\/a><\/span>',re.S|re.I).findall(chunk)
            if(len(matches)>0):
                name=matches[0]
                
            #Parsing rating
            matches=re.compile(r'(\d.0) of 5 bubbles',re.S|re.I).findall(chunk)
            if(len(matches)>0):
                rating=matches[0].strip()
                
            #parsing user location
            matches=re.compile(r'osNWb\"><span>(.*?)<\/span>',re.S|re.I).findall(chunk)
            if(len(matches)>0):
                user_location=matches[0].strip()
            
            #parsing the contribution
            matches=re.compile(r'<span class=\"\">(.*?) contribution',re.S|re.I).findall(chunk)
            if (len(matches)==0): 
                matches=re.compile(r'\"IugUm\">(.*?) contribution',re.S|re.I).findall(chunk)
            contribution=matches[0].strip()
            
            #parsing the month and year
            matches=re.compile(r'RpeCd\">(.*?) (\d{4})',re.S|re.I).findall(chunk)
            if(len(matches) == 1):
                month = matches[0][0].strip()
                year = matches[0][1].strip()
        
            
            #parsing the message 
            matches=re.compile(r'<span class="yCeTE">(.*?)<\/span>',re.S|re.I).findall(chunk)
            if(len(matches)>0):
                review_message=matches[0].strip()
            
            #printing collected data to screen
            print(name+":"+rating+":"+contribution+":"+user_location+":"+review_message+":"+month+":"+year)
             #Save the extracted data into the database
            query = "INSERT INTO Tripadvisorreviewstable VALUES (? ,? , ?, ?, ?, ?, ?)"
            c.execute(query, (name, rating, contribution, user_location, review_message, month, year))
 
    #Is there a next page?
    URL_pattern_str = "https://www.tripadvisor.com/AttractionProductReview-g60745-d11450485-or"+str(n)+"-Boston_Hop_On_Hop_Off_Trolley_Tour_with_14_Stops-Boston_Massachusetts.html"
    page_URL = URL_pattern_str
    print("Collecting data from" + page_URL)
    time.sleep(2)
    browser.get(page_URL)
    page_content=browser.page_source

    n = n + 10


conn.commit()
conn.close()

browser.close()
print("\n\nCollection Finished!")  


# In[3]:


import sqlite3

conn = sqlite3.connect('group.db')
c = conn.cursor()

c.execute("SELECT * FROM Tripadvisorreviewstable ")
result = c.fetchall()

print(result)

c.close()
conn.close()


# In[4]:


import sqlite3
conn = sqlite3.connect("group.db")
cursor = conn.cursor()
cursor.execute ("SELECT * from Tripadvisorreviewstable ")
rows =cursor.fetchall()
header = ""
for column_info in cursor.description:
    header += column_info[0] + ","
print(header)
for row in rows:
    print(row) 
conn.commit()
cursor.close()
conn.close ()


# In[5]:


import pandas as pd
df = pd.DataFrame(rows, columns = ['Name','rating', 'Contribution', 'userLocation', 'reviewMessage', 'month','year'])
print(df)


# In[6]:


import numpy as np
#def recode_empty_cells(df,['userLocation','month','year']):
for column in ['userLocation','month','year']:
    df[column] = df[column].replace(r'^\s*$', np.nan, regex=True)
    df[column] = df[column].fillna(0)
    
print(df)


# In[7]:


df.to_csv("tripadvisortable.csv") 
Tripadvisor_boston =pd.read_csv('tripadvisortable.csv')
Tripadvisor_boston


# In[8]:


Tripadvisor_boston.describe(include='all')


# In[9]:


Tripadvisor_boston.info()


# In[10]:


Tripadvisor_boston.shape
#1171 observations and 8 characteristics


# In[11]:


Tripadvisor_boston['Name'].value_counts()


# In[12]:


Tripadvisor_boston['rating'].value_counts()


# In[13]:


Tripadvisor_boston['Contribution'].value_counts()


# In[14]:


Tripadvisor_boston['userLocation'].value_counts()


# In[15]:


Tripadvisor_boston['reviewMessage'].value_counts()


# In[16]:


Tripadvisor_boston['month'].value_counts()


# In[17]:


Tripadvisor_boston['year'].value_counts()


# In[18]:


Tripadvisor_boston.groupby('Contribution').describe()


# In[19]:


Tripadvisor_boston2 = Tripadvisor_boston.groupby(['userLocation','year'])['userLocation'].count()
Tripadvisor_boston2


# In[20]:


pip install TextBlob


# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from textblob import TextBlob


# In[22]:


sns.countplot(x='rating',data=Tripadvisor_boston)


# In[23]:


#Visualization
ax = plt.subplots(figsize=(15,45))
sns.scatterplot(x = "rating", y = "Contribution", data=Tripadvisor_boston)
plt.show()


# In[24]:


Tripadvisor_boston1 = Tripadvisor_boston['year'] > 0
Tripadvisor_boston1


# In[25]:


Tripadvisor_boston7 = Tripadvisor_boston[Tripadvisor_boston1]
Tripadvisor_boston7


# In[26]:


ax = plt.subplots(figsize=(8,10))
plt.hist(Tripadvisor_boston7['year'])
plt.title("Histogram")
plt.show()


# In[27]:


Tripadvisor_boston3 = Tripadvisor_boston.groupby(['year','rating'])['reviewMessage'].agg('count').reset_index()
Tripadvisor_boston3


# In[28]:


Tripadvisor_boston4=Tripadvisor_boston3.iloc[5:31,]
Tripadvisor_boston4


# In[29]:


plt.figure(figsize=(8,10))
sns.scatterplot(x = 'rating', y = 'reviewMessage', hue = 'year', data = Tripadvisor_boston4);


# In[30]:


Tripadvisor_boston5 = Tripadvisor_boston.sort_values(
     by=["Contribution", "Name"],
    ascending=False
   )[["Contribution", "Name"]]

Tripadvisor_boston5


# In[31]:


Tripadvisor_boston6 = Tripadvisor_boston5.head(20)
Tripadvisor_boston6


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns


fig, ax = plt.subplots(figsize=(15,10))
sns.scatterplot(data=Tripadvisor_boston6.sort_values(
     by=["Contribution", "Name"],
    ascending=True
   )[["Contribution", "Name"]], x="Contribution", y="Name", size="Contribution", legend=False, sizes=(2000, 30))
plt.show()


# In[34]:


with open("reviewMessage.txt", "w") as f_out:
    f_out.write(" ".join(Tripadvisor_boston["reviewMessage"].str.lower()))


# In[35]:


#open text file in read mode
text_file = open("reviewMessage.txt", "r")
 
#read whole file to a string
text = text_file.read()
text


# In[36]:


def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[37]:


# Import package
from wordcloud import WordCloud, STOPWORDS
# Generate word cloud
wordcloud = WordCloud(width= 3000, height = 2000, random_state=1, background_color='black', colormap='Pastel1', collocations=False, stopwords = STOPWORDS).generate(text)
# Plot
plot_cloud(wordcloud)


# In[38]:


conn = sqlite3.connect('group.db')
cursor = conn.cursor()
query = 'SELECT * FROM Tripadvisorreviewstable'
DF = pd.read_sql(query, conn)
DF


# In[39]:


DF = DF[DF.year != '']
DF = DF[DF.rating != '']
DF = DF[DF.reviewMessage != '']
#DF = DF[DF.contribution != '']
DF['year'] =DF['year'].astype(int)
DF['rating'] =DF['rating'].astype(float)
DF['reviewMessage'] =DF['reviewMessage'].astype(str)
#DF['contribution'] =DF['contribution'].astype(float)
DF['reviewMessage'].dtypes


# In[44]:


import numpy as np 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
y=DF[['rating']].to_numpy()
x=DF[['year']].to_numpy()
plt.figure(figsize = (10,5))
model = LinearRegression()
print(model.fit(x, y))
print(model.intercept_)
print("Coefficient is = ", model.coef_)
print("R-Square is = ", model.score(x, y))


y_est = model.predict(x)
ax=plt.axes()
ax.set_xlabel('rating')
ax.set_ylabel('year')
plt.xlim(2016,2023)
ax.scatter(x, y)
ax.plot(x,y_est)
x=sm.add_constant(x)
model_ols=sm.OLS(y,x)
results=model_ols.fit()
print(results.summary())


x_new = np.asarray([2019])
y_pred=model.predict(x_new.reshape(1,-1)) 
print("Estimated ratings for the new data:", y_pred)
ax.scatter(x_new,y_pred,color='red')


# In[45]:


#sentiment Analysis
def find_pol(review):
    return TextBlob(review).sentiment.polarity

DF['Sentiment_Polarity'] = DF['reviewMessage'].apply(find_pol)
DF.head()


# In[46]:


#polarity test
#the histogram show that the majority of review is positive that means they are good review 
sns.distplot(DF['Sentiment_Polarity'])


# In[47]:


sns.barplot(x='rating', y='Sentiment_Polarity',data=DF)


# In[48]:


most_negative = DF[DF.Sentiment_Polarity == -1].reviewMessage.head()
print(most_negative)


# In[49]:


most_positive = DF[DF.Sentiment_Polarity == 1].reviewMessage.head()
print(most_positive)

