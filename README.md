# Ex02-Outlier

### AIM

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them


### EXPERIMENT

An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.


### ALGORITHM

1. Read the given Data
2. Get the information about the data
3. Detect the Outliers using IQR method and Z score
4. Remove the outliers
5. Plot the datas using Box Plot

### PROGRAM 

#### (1) & (2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe

```

import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(bhp.csv")
df

df.head()

df.describe()

df.info()

df.isnull().sum()

df.shape

sns.boxplot(x="price_per_sqft",data=df)

q1 = df['price_per_sqft'].quantile(0.25)
q3 = df['price_Aper_sqft'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df1 =df[((df['price_per_sqft']>=ll)&(df['price_per_sqft']<=ul))]
df1

df1.shape

sns.boxplot(x="price_per_sqft",data=df1)
```

#### (3) Examine price_per_sqft column and use zscore of 3 to remove outliers.

```
from scipy import stats

z = np.abs(stats.zscore(df['price_per_sqft']))
df2 = df[(z<3)]
df2

print(df2.shape)
sns.boxplot(x="price_per_sqft",data=df2)
```

#### (4)(i) For the data set height_weight.csv detect weight outliers using IQR method

```
df3 = pd.read_csv("height_weight.csv")
df3

df3.head()

df3.info()

df3.describe()

df3.isnull().sum()

df3.shape

sns.boxplot(x="weight",data=df3)

q1 = df3['weight'].quantile(0.25)
q3 = df3['weight'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df4 =df3[((df3['weight']>=ll)&(df3['weight']<=ul))]
df4

df4.shape

sns.boxplot(x="weight",data=df4)
```

#### (4)(ii) For the data set height_weight.csv detect height outliers using IQR method

```
sns.boxplot(x="height",data=df3)

q1 = df3['height'].quantile(0.25)
q3 = df3['height'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df5 =df3[((df3['height']>=ll)&(df3['height']<=ul))]
df5

df5.shape

sns.boxplot(x="height",data=df5)
```
### Output 

#### (1)(2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe

![1-1](https://user-images.githubusercontent.com/93901857/190849137-271b8e19-2326-4ba2-a376-9e68d5604e8e.jpg)
![1-2](https://user-images.githubusercontent.com/93901857/190849140-1b7822a3-8a03-4b13-9928-adb5ee5f74a7.jpg)
![1-3](https://user-images.githubusercontent.com/93901857/190849142-e94c90e9-a6b1-45f6-802b-fd04ab013057.jpg)
![1-4](https://user-images.githubusercontent.com/93901857/190849143-aa5abb12-595a-4b90-bcaf-fcb3f29d4341.jpg)
![1-5](https://user-images.githubusercontent.com/93901857/190849145-b8f13265-060e-4cc7-8df1-7a6b18b03b3e.jpg)
![1-6](https://user-images.githubusercontent.com/93901857/190849147-a50c058b-154d-4f89-85f0-0962716ee70c.jpg)

#### (3) Examine price_per_sqft column and use zscore of 3 to remove outliers.

![3-1](https://user-images.githubusercontent.com/93901857/190849309-6bbddb87-4a49-4198-be2e-ed60fb1e9568.jpg)



#### (4) For the data set height_weight.csv detect weight and height outliers using IQR method



![3-2](https://user-images.githubusercontent.com/93901857/190849554-bbe3faba-b2cd-4b4c-a31d-b63b58d0d310.jpg)
![3-3](https://user-images.githubusercontent.com/93901857/190849556-293a226f-188c-45a9-999c-c4a1c136ef3c.jpg)
![3-4](https://user-images.githubusercontent.com/93901857/190849557-65e67920-cb08-48ce-ae59-25fed620f10a.jpg)
![3-5](https://user-images.githubusercontent.com/93901857/190849558-0a5a804c-c73d-4eb9-aae0-b4bdf88db05e.jpg)

![3-6](https://user-images.githubusercontent.com/93901857/190849462-729ee208-fa04-492a-9dab-2eba946bdc51.jpg)
![3-7](https://user-images.githubusercontent.com/93901857/190849463-aefc94b3-f54f-4d82-bf98-759d90d06b18.jpg)
![3-8](https://user-images.githubusercontent.com/93901857/190849464-247e60d9-81ce-481b-8bf6-23fdac3eb76e.jpg)

![3-9](https://user-images.githubusercontent.com/93901857/190849574-505bb5d1-8bfe-4c51-86a8-727fcfef9bb4.jpg)
![3-10](https://user-images.githubusercontent.com/93901857/190849576-5dcc7b18-3774-4806-8c10-7e27bc68cc4a.jpg)


