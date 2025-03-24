<H3>ENTER YOUR NAME</H3> NAVEEN KUMAR B
<H3>ENTER YOUR REGISTER NO.</H3> 212222230091
<H3>EX. NO.1</H3>
<H3>DATE</H3>7.3.25
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#import libraries
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#Read the dataset from drive
df=pd.read_csv("/content/Churn_Modelling.csv")
df

df.isnull().sum()

#check for duplication
df.duplicated()

print(df['CreditScore'].describe())

df.info()

df.drop(['Surname','Geography','Gender'],axis=1,inplace=True)
df

Scaler=MinMaxScaler()
df1=pd.DataFrame(Scaler.fit_transform(df))
df1

X = df1.iloc[:, :-1].values
print(X)

y = df1.iloc[:,-1].values
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=25)

print(X_train)
print(len(X_train))

print(X_test)
print(len(X_test))
```


## OUTPUT:
![Screenshot 2025-03-07 175552](https://github.com/user-attachments/assets/0ffce58a-f05c-44b1-84e5-00248c90b114)


![Screenshot 2025-03-07 175627](https://github.com/user-attachments/assets/005fa127-0124-4cb2-958a-d85a657e413c)


![Screenshot 2025-03-07 180301](https://github.com/user-attachments/assets/d00e28b3-7961-4942-a514-899d99f1aed1)


![Screenshot 2025-03-07 180306](https://github.com/user-attachments/assets/1197f4b0-a8b9-4d8a-972a-538ed490bbaa)


![Screenshot 2025-03-07 180311](https://github.com/user-attachments/assets/f27264fe-bdc9-4d83-8014-92f2d7204a19)


![Screenshot 2025-03-07 180320](https://github.com/user-attachments/assets/628baeca-a721-4e3e-a118-561799024342)


![Screenshot 2025-03-07 181120](https://github.com/user-attachments/assets/350eb9e3-65e1-4728-a912-1a11b9121c64)


![Screenshot 2025-03-07 180344](https://github.com/user-attachments/assets/8022088c-df2f-4343-89bf-d8f3596fee74)


![Screenshot 2025-03-07 180351](https://github.com/user-attachments/assets/4fb3aa50-582b-4fb4-8c6c-f40489ecdc3b)


![Screenshot 2025-03-07 180401](https://github.com/user-attachments/assets/bec0cfb2-da2e-4fa0-a138-d9591e195219)


![Screenshot 2025-03-07 180410](https://github.com/user-attachments/assets/8524a558-f8fe-4f25-9317-9e1d1a01c680)

![Screenshot 2025-03-07 181328](https://github.com/user-attachments/assets/310323b5-c36e-46d4-8d79-d1340ee20a71)
















## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


