---
title: "Heart diseases"
output: html_notebook
---

# Heart diseases

### Our goal:

The goal of collecting a dataset on heart diseases is to gather relevant information about the factors that influence the risk of gaining heart diseases. The results of analyzing this dataset will be utilized in various beneficial ways, research advancements, researchers work towards improving the usnderstanding and management of heart disease, and increasing awareness among the public regarding the common factors that can contribute to heart diseases.

### Classification goal:

The aim is to predict whether a person is at risk of having a certain type or level of heart disease based on their vital signals, such as age, sex, blood pressure, and cholesterol levels. The objective is to build a predictive model that accurately categorizes individuals into two groups: those likely to have a future heart disease (labeled as 1) and those unlikely to have a future heart disease (labeled as 0).

### Clustring goal:

The aim is to divide the people into several groups based on their information in the attributes values. The division is based on the similarities and differences between features without knowing a class label.

### Defect prediction goal:

The objective is to develop a model that can accurately classify individuals into categories based on the presence or absence of a specific heart defect. This can aid in early detection, diagnosis, and treatment planning. Identifying these defects early allows healthcare professionals to intervene promptly and provide appropriate medical interventions.

Overall, our DataSet for heart disease provides healthcare professionals with a powerful tool for diagnosing, classifying, and predicting heart disease.

#### Source:
Kaggle website

#### URL:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset


### Data Type:
Our dataset has 14 objects (attributes).
our class label is the target. It is binary type which has value 0= no disease , 1=disease.

|Attributes name|Description|Data type|Possible values|
|---------------|-----------|---------|---------------|
|Age|The person's age in years|Numeric|from 29 to 77|
|sex|The person gender (Female,Male)|Binary| 1=male , 0=female|
|cp|chest pain type|Nominal|from 0 to 3 which 0=asymptomatic , 1= atypical angina , 2= non-anginal pain , 3= typical angina|
|trestbps|resting blood pressure (in mm Hg on admission to the hospital)|Numeric|from 94 to 200|
|chol|serum cholestoral in mg/dl| Numeric| from 126 to 564|
|fbs|fasting blood sugar greater than 120 mg/dl|Binary| 1=true , 0=false|
|restecg|resting electrocardiographic results|Nominal|from 0 to 2 which 0=showing probable or definite left ventricular hypertrophy by Estes’ criteria , 1=normal , 2=having ST-T wave abnormality|
|thalach|maximum heart rate achieved|Numeric| from 71 to 202|
|exang|exercise induced angina|Binary|1=yes , 0=no|
|oldpeak|ST depression induced by exercise relative to rest|Numeric|from 0 to 6.2|
|slope|the slope of the peak exercise ST segment|Ordinal| from 0 to 2 which 0=downsloping , 1=flat , 2=upsloping|
|ca|The number of major vessels(0-3)colored by flourosopy| Nominal | from 0 to 3|
|thal|A blood disorder called thalassemia Value|Nominal| from 1 to 3 which 1=normal , 2= fixed defect , 3=reversible defect|

### Importing libraries
```{r}
install.packages("ggplot2")
library(mlbench)
library(aod)
library(mlbench)
library(dplyr)
library(ggplot2)
library(pROC)
library(stats)
library(rpart.plot)
library(randomForest)
library(gbm)
library(class)
library(patchwork)
library(e1071)
library(caret)
```
### Reading the data:
```{r}
data= read.csv('heart.csv')
head(data)
```
### Check the duplication:
To make sure that our data is clean and doesn't contain redundant data we start by using the code that help us to find the duplication. 
```{r}
sum(duplicated(data))
```
From the previous output we notice that we have 723 duplicated objects. For Example:
```{r}
data[c(61,65,119,669),]
```
Therefor, we decided to remove the duplicated data, which result in having 302 different object in our new data set, which is shown below.

```{r}
dataset <- data[!duplicated(data), ]
dataset
```
```{r}
sum(duplicated(dataset))
```
After removing the duplicated data the sum of the duplicated objects is 0, so we made sure that our objects are distinct.

## Number of rows and coloumns:
To represent our new data set columns and rows we use this code. It shows that our new data set consists of 14 coloumns and 302 rows.
```{r}
ncol(dataset)
nrow(dataset)
```

## Statistical measures:
We will represent the five number summary, the mean and the variance for the numerical attributes. These values will give us an overview about our attributes. Also, it could be useful for smoothing and handling the outliers, missing and wrong values.
#### For the Age attribute:
```{r}
summary(dataset$age)
var(dataset$age)
```
We notice that our data set patients age range from 29 to 77 years old, which indicate that all of them are middle aged and old adults.

#### For the resting blood pressure (trestbps) attribute:
```{r}
summary(dataset$trestbps)
var(dataset$trestbps)
```
We notice that our data set patients resting blood pressure range from 94 to 200. 

#### For the serum cholestoral (chol) attribute:
```{r}
summary (dataset$chol)
var(dataset$chol)
```
We notice that our data set patients serum cholestoral range from 126 to 564.
#### For the maximum heart rate achieved (thalach) attribute:
```{r}
summary(dataset$thalach)
var(dataset$thalach)
```
We notice that our data set patients maximum heart rate achieved range from 71 to 202.
#### For the ST depression (oldpeak) attribute:
```{r}
summary(dataset$oldpeak)
var(dataset$oldpeak)
```
We notice that our data set patients maximum heart rate achieved range from 0 to 6.2.

### Boxplot:
We use boxplot to provide us with a quick visual summary by showing graphical representation of the five number summary for each numeric attribute. Also, it help us to know and detect the outliers from graph.
#### For the Age attribute:
```{r}
boxplot(dataset$age)
```
The age summary and boxplot show that all people in our data set have ages ranging from 29 to 77.There are no outliers in age attribute. The median (55.50) is close to the 3rd quartile (61.00) indicating a slightly skewed distribution.

#### For the resting blood pressure (trestbps) attribute:
```{r}
boxplot(dataset$trestbps)
```
The trestbps summary and boxplot show that resting blood pressure values (trestbps) span from 94mmHg to 200mmHg. There are few outliers, that we will deal with in preprocessing section. The distribution of the data appears to be symmetric where the median (130.0mmHg) line is centered within the box, and the whiskers are of similar length on both sides.

#### For the serum cholestoral (chol) attribute:
```{r}
boxplot(dataset$chol)
```
The serum cholesterol (chol) summary and boxplot shows that the attribute range from 126 to 564. We notice from the graph that there are few outliers where all of them are close to each other except one of them. As we mention before we will handle the outliers in preprocessing section.

#### For the maximum heart rate achieved (thalach) attribute:
```{r}
boxplot(dataset$thalach)
```
The summary and boxplot of maximum heart rate (thalach) data reveal that the maximum heart rate values range from 71.0 to 202.0. The boxplot visually presents the distribution of the attribute data appears to be slightly skewed distribution.

#### For the ST depression (oldpeak) attribute:
```{r}
boxplot(dataset$oldpeak)
```
The summary and boxplot of ST depression (oldpeak) attribute range from 0 to 6.2. Also, it displays only the box and the upper whisker and it does not have a lower whisker. That means that there are no values below the lower fence.

### Graphical representations:
#### Pie chart for our class label (target).
```{r}
tab <- dataset$target %>% table()
percentages <- tab %>% prop.table() %>% round(3) * 100
txt <- paste0(names(tab), '\n', percentages, '%')
pie(tab, labels = txt, main="percentage of the target")
```
The pie chart help us to know the percentage of the people in our data set who might be targeted by heart diseases. More than half of the people (represented by 54.3%) has high potentials of getting infected. In other side less than half of the people (represented by 45.7%) has low potentials of getting infected. So, we determine that our data set is balanced since the two percentage of our class label are almost close together.

#### Graph between the sex and the target bar chart.
```{r}
ggplot(dataset, aes(x = sex, fill = as.factor(target))) +
  geom_bar(stat = "count", position = "stack", width = 0.6, show.legend = TRUE) +
  geom_text(aes(label = after_stat(count)), stat = 'count', position = position_stack(vjust = 0.5)) +
  labs(x = "Sex", y = "Count",fill="target") 
```
The chart represent sex attribute and our class label (target). Most of the people in our data set were male. For male, the target is almost equally distributed. For female, the majority of them are considered targeted. 

#### Graph between the chest pain type (cp) and the target.
```{r}
ggplot(dataset, aes(x = cp, fill = as.factor(target))) +
  geom_bar(stat = "count", position = "stack", width = 0.6, show.legend = TRUE) +
  geom_text(aes(label = after_stat(count)), stat = 'count', position = position_stack(vjust = 0.5)) +
  labs(x = "cp", y = "Count",fill="target") 
```
The chart represent chest pain type (cp) attribute and our class label (target).The chart shows that "0" which indicate asymptomatic type are people who don't have high potentiality to be infected. In contrast, The other three types ("1" atypical angina ,"2" non-anginal pain ,"3" typical angina) have higher potential to be infected.

#### Graph between the number of major vessels (ca) and the target.
```{r}
ggplot(dataset, aes(x = ca, fill = as.factor(target))) +
  geom_bar(stat = "count", position = "stack", width = 0.6, show.legend = TRUE) +
  geom_text(aes(label = after_stat(count)), stat = 'count', position = position_stack(vjust = 0.5)) +
  labs(x = "ca", y = "Count",fill="target")
```
The chart represent the number of major vessels colored by flourosopy (ca) attribute and our class label (target). The graph shows that ca values are correlated to the possibility of getting infected. Where for vessel "0" approximately three quarters of people are infected.In contrast "1,2,3" vessels show that the targeted people have less percentage of getting infected. Also, we detected a wrong value which is 4 but, we will handle it in the preprocessing section.

#### Graph between maximun heart rate (thalach) and the target.
```{r}
plot(dataset$target,dataset$thalach)
```
The scatter plot represent maximum heart rate achieved (thalach) attribute and our class label (target). We notice from the graph that the majority of targeted people has high thalach value (thalach>130). 

### Scatter Plot for Age and resting blood sugar (trestbps).
```{r}
plot(dataset$age, dataset$trestbps, pch = 16, col = "black", xlab = "Age", ylab = "trestbps", main = "Scatter Plot for Age and trestbps")
abline(lm(dataset$trestbps ~ dataset$age), col = "red", lwd = 2)
```
The scatter plot represent resting blood pressure (trestbps) and age attributes. The line shows that there is a slight positive correlation between them. Also, we noticed that trestbps has high values between 50 to 60 years.

### Scatter Plot for Maximum heart rate (thalach) and ST depression induced by exercise (oldpeak).
```{r}
plot(dataset$thalach, dataset$oldpeak, pch = 16, col = "black", xlab = "thalach", ylab = "oldpeak", main = "Scatter Plot for thalach and oldpeak")
abline(lm(dataset$oldpeak ~ dataset$thalach), col = "red", lwd = 2)
```
The scatter plot represent ST depression induced by exercise relative to rest (oldpeak) and a blood disorder called thalassemia Value (thalach) attributes. The line shows that there is a negative correlation between them. Also, we noticed that the most frequent value for oldpeak is zero which was centrelized between 150 to 185.
```{r}
plot(dataset$chol,dataset$target)
```
The scatter plot represent serum cholestoral (chol) and our class label (target) attributes. We think that there is a very weak correlation between them since, the representation of 0 and 1 in y axis are almost similar. We will prove that when we apply the correlation coefficient in the preprocessing section.

#### Graph between the fasting blood sugar(fbs) and the target.
```{r}
ggplot(dataset, aes(x = fbs, fill = as.factor(target))) +
  geom_bar(stat = "count", position = "stack", width = 0.6, show.legend = TRUE) +
  geom_text(aes(label = after_stat(count)), stat = 'count', position = position_stack(vjust = 0.5)) +
  labs(x = "fbs", y = "Count",fill="target")
```
The chart represent the fasting blood sugar(fbs) (1 means grater than 120 and 0 means less than 120) attribute and our class label (target). From the graph, Either fbs is less or greater than 120 they both have the same percentage of getting infected which is approximately 50% so, we determine that they are not correlated. We will prove that when we apply the correlation coefficient in the preprocessing section.

### Preprocessing:
In this section, we will apply preprocessing techniques which will help us to clean and remove noise from our data set in order to get better and more accurate results. We will perform some methods, Starting from managing wrong, missing and outliers values then applying discretization, normlization and encoding. Also, perform feature selection and finding the correlation between the attributes.

### Detecting the missing values: 
```{r}
sum(is.na(data))
```
since the output is 0 so, there is no missing values in our data set.

### Detecting outliers:

#### for the Age attributes:
```{r}
boxplot.stats(dataset$age)$out
```
There is no outliers in the age attribute.

#### for the resting blood pressure (trestbps) attributes:
```{r}
boxplot.stats(dataset$trestbps)$out
```
From the previous code we notice that we have 9 outliers but, the pressure can be over 180 in some people which considered high level of pressure so we will not consider it as an outlier.

#### for the serum cholestoral (chol) attributes:
```{r}
boxplot.stats(dataset$chol)$out
```
From the previous code we notice that we have 5 outliers but, 4 of them were close to the rest of the serum cholestoral values except 564 so we will handle it by smoothing using mean of the value in the following steps. 

#### for the maximum heart rate achieved (thalach) attributes:
```{r}
boxplot.stats(dataset$thalach)$out
```
From the previous code we notice that we have one outlier but, we will not consider it as an outlier since it's medically acceptable for the maximum heart rate achieved.

#### for the ST depression (oldpeak) attributes:
```{r}
boxplot.stats(dataset$oldpeak)$out
```
From the previous code we notice that we have 5 outliers but, we will not consider them as outliers since these values are medically acceptable for the ST depression.

### Handling outliers values:
We have one outlier that we will handle which is in serum cholestoral (chol) attributes with the value 564. Since it's only one value we're going to handle it manually by replacing it with the mean. We extracted the mean value from the summary code.
```{r}
dataset[129,5]=246.5
```
### Detecting wrong values:
We are going to check if there is any wrong values out of the attributes values range.

#### The number of major vessels (ca) attribute:
```{r}
dataset[dataset$ca != "3" & dataset$ca != "1" & dataset$ca != "2" & dataset$ca != "0",]
```
We noticed that value number 4 is considered as a wrong value. It has been in 4 rows. We will handle it in the following step.

#### Sex attribute:
```{r}
dataset[dataset$sex != "0" & dataset$sex!= "1",]
```
There's no wrong value in the sex attribute.

#### Fasting blood sugar (fbs) attribute:
```{r}
dataset[dataset$fbs != "0" & dataset$fbs != "1",]
```
There's no wrong value in the fasting blood sugar attribute.

#### Resting electrocardiographic results (restecg) attribute:
```{r}
dataset[dataset$restecg != "0" & dataset$restecg != "1" & dataset$restecg != "2" ,]
```
There's no wrong value in the Resting electrocardiographic results attribute.

#### Exercise induced angina (exang) attribute:
```{r}
dataset[dataset$exang != "0" & dataset$exang != "1",]
```
There's no wrong value in the Exercise induced angina attribute.

#### The slope of the peak exercise ST segment (slope) attribute:
```{r}
dataset[dataset$slope != "0" & dataset$slope != "1" & dataset$slope != "2" ,]
```
There's no wrong value in the slope of the peak exercise ST segment attribute.

#### A blood disorder (thal) attribute:
```{r}
dataset[dataset$thal != "1" & dataset$thal != "2" & dataset$thal != "3" ,]
```
We noticed that value number 0 is considered as a wrong value. It has been in 2 rows. We will handle it in the following step.

#### Result for detecting wrong values: 
We notice that number of major vessels and blood disorder attributes has wrong values (out of range).These values will be handled by using central tendencies values.

### Handling wrong values:
- The number of major vessels (ca) attribute:
```{r}
getmode <- function(v) {
   uniqv <- unique(v)
   uniqv[which.max(tabulate(match(v, uniqv)))]}
a=getmode(dataset$ca)
print(a)
dataset[dataset$ca=="4","ca"]<-a
```
We replaced the wrong value with the attribute mode since it's type is nominal.


- A blood disorder (thal) attribute:
```{r}
getmode <- function(v) {
   uniqv <- unique(v)
   uniqv[which.max(tabulate(match(v, uniqv)))]}
b=getmode(dataset$thal)
print(b)
dataset[dataset$thal=="0","thal"]<-b
```
We replaced the wrong value with the attribute mode since it's type is nominal.

### Discretization: 
We will use discretization for resting blood pressure (trestbps) and age attributes which help us to form intervals and each interval has categorical label.

#### Resting blood pressure (trestbps):
```{r}
dataset$trestbps <- ifelse(dataset$trestbps <=90, "Low",
                           ifelse(dataset$trestbps <= 120, "Normal",
                                 ifelse(dataset$trestbps <= 129, "Elevated",
                                        ifelse(dataset$trestbps <= 139, "High Stage 1",
                                                ifelse(dataset$trestbps <= 180, "High Stage 2",
                                                     ifelse(dataset$trestbps > 180, "High Stage 3",0))))))

trestbpsdis <- dataset$trestbps
print(trestbpsdis)
```
#### Age:
```{r}
AgeBeforeDis <- dataset$age
```

```{r}
dataset$age <- ifelse(dataset$age <= 16, "Children",
                      ifelse(dataset$age <= 39, "Young Adults",
                                 ifelse(dataset$age <= 59, "Middle-aged Adults",
                                        ifelse(dataset$age < 99, "Old Adults",0))))
agedis <- dataset$age
print(agedis)
```

### Normalization: 
The normalization step will transform the attributes values into smaller range which will help us to provide an equal weight for the attributes.
We will use min-max normalization for many attributes.

#### Call min-max normalize function
```{r}
min_max_normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))}
```

#### Applying normlization function in serum cholestoral (chol):
```{r}
dataset$chol <- min_max_normalize(dataset$chol)
print(dataset$chol)
```
#### Applying normlization function in ST depression induced by exercise (oldpeak):
```{r}
dataset$oldpeak <- min_max_normalize(dataset$oldpeak)
print(dataset$oldpeak)
```

#### Applying normlization function in maximum heart rate achieved (thalach):
```{r}
dataset$thalach <- min_max_normalize(dataset$thalach)
print(dataset$thalach)
```


### Encoding:
Most of our attributes values are already encoded in the data set. We will encode the age and resting blood pressure (trestbps).

For example the sex is encoded to 0's and 1's where 0= female and 1= male as a binary type. What's shown below are the values of sex column.
```{r}
sexencoding <- dataset$sex
print(sexencoding)
```
#### Encoding for resting blood pressure (trestbps):
```{r}
dataset$trestbps=factor(dataset$trestbps,levels=c("Low","Normal","Elevated","High Stage 1","High Stage 2","High Stage 3"),labels=c(0,1,2,3,4,5))
```

#### Encoding for age:
```{r}
dataset$age=factor(dataset$age,levels=c("Children","Young Adults","Middle-aged Adults","Old Adults"),labels=c(0,1,2,3))
```


## Correlation Analysis:
We will find the correlation between each attributes and the class label (target). For the nominal data we will use chi-square and for the numeric data we will use correlation coefficient. This will help us to determine the most important and correlated attributes to the target.

### Chi-square for nominal data:

  |Attribute name | Chi-square value|Degree of freedom|Alpha|             
  |-----------------|-----------------|-----------------|-----------------|
  | A blood disorder (thal) | 83.978|2|2.2e-16|
  | Chest pain type (cp)    | 80.979  |3| 2.2e-16| 
  | Number of major vessels (ca)| 73.689 |3|6.919e-16|
  | Exercise induced anginal (exang)| 55.456 |1|9.556e-14|
  |The slope of the peak exercise ST segment (slope)| 46.889|2|6.578e-11|
  | Sex            | 23.084       | 1| 1.551e-06|
  |Resting blood pressure (trestbps)|  9.8824 |4|0.04246|
  | Resting electrocardiographic result (restecg)| 9.7297 |2|0.007713|     
  |Age(after discretization)| 8.7992  |2|0.01228|      
  | Fasting blood sugar (fbs)    | 0.092408               | 1|0.7611|                                     
  

#### Sex:
```{r}
csex=chisq.test(dataset$sex , dataset$target)
print(csex)
```
#### Chest pain type (cp):
```{r}
ccp=chisq.test(dataset$cp , dataset$target)
print(ccp)
```
#### Fasting blood sugar (fbs):
```{r}
cfbs=chisq.test(dataset$fbs , dataset$target)
print(cfbs)
```
#### Resting electrocardiographic result (restecg):
```{r}
crestecg=chisq.test(dataset$restecg , dataset$target)
print(crestecg)
```
#### Exercise induced anginal (exang):
```{r}
cexang=chisq.test(dataset$exang , dataset$target)
print(cexang)
```
#### The slope of the peak exercise ST segment (slope):
```{r}
cslope=chisq.test(dataset$slope , dataset$target)
print(cslope)
```
#### Number of major vessels (ca):
```{r}
cca=chisq.test(dataset$ca , dataset$target)
print(cca)
```
#### A blood disorder (thal):
```{r}
cthal=chisq.test(dataset$thal , dataset$target)
print(cthal)
```
#### Age:
The age attribute after discretization.
```{r}
cage= chisq.test(dataset$age, dataset$target)
print(cage)
```
#### Resting blood pressure (trestbps):
The trestbps attribute after discretization.
```{r}
ctrestbps=chisq.test(dataset$trestbps ,dataset$target)
print(ctrestbps)
```
### Chi-square Results:
We will sort the Chi-square values from the highest to the lowest:
1) A blood disorder (thal) with value 83.978.
2) Chest pain type (cp) with value 80.979.
3) Number of major vessels (ca) with value 73.69.
4) Exercise induced anginal (exang) with value 55.456.
5) The slope of the peak exercise ST segment (slope) with value 46.889.
6) Sex with value 23.084.
7) resting blood pressure (trestbps) with value 9.8824.
8) Resting electrocardiographic result (restecg) with value 9.7297.
9) Age with value 8.7992.
10) Fasting blood sugar(fbs) with value 0.092408.

All of the attributes is dependent to the class lable (target) except Fasting blood sugar(fbs) since the p-value is higher than chi-square value (chi-square = 0.092408 < 0.7611 = p-value).


### Correlation coefficient for numeric data:

#### Serum cholestoral (chol):
```{r}
cchol=cor(dataset$chol ,dataset$target)
print(cchol)
```
#### Maximum heart rate achieved (thalach):
```{r}
cthalach=cor(dataset$thalach ,dataset$target)
print(cthalach)
```
#### ST depression indicated by exercise relative to rest (oldpeak):
```{r}
coldpeak=cor(dataset$oldpeak ,dataset$target)
print(coldpeak)
```
### Correlation coefficient Results:
The highest correlation coefficient value is ST depression indicated by exercise relative to rest (oldpeak) which is negatively correlated (-0.4291458) with the target. Then, the value of maximum heart rate achieved (thalach) which is positively correlated (0.419955) with the target. Lastly, the value of Serum cholestoral (chol) which is negatively correlated (-0.1070417) with the target.
 
From the previous results we decided to delete the Fasting blood sugar (fbs) since it is independent from the class label (target) which means that fasting blood sugar (fbs) doesn't affect our class label (target).

```{r}
dataset <- dataset[, -which(names(dataset) == "fbs")]
head(dataset)
```
```{r}
ncol(dataset)
nrow(dataset)
```
After printing a sample from our new data set we notice that the attribute decrease from 14 to 13 since we remove Fasting blood sugar (fbs). Also, the rows decrease from 1025 to 302 since we remove the redunant data. 

###### Let the target be binary:
```{r}
dataset$target <- as.factor(dataset$target)
```

## Classification and Clustering:
We will apply supervised (Classification) and unsupervised (Clustering) learning techniques on our data set.

### Classification:
It's a process about labeling the data object into predefined classes based on their attribute values. It aims to build and construct a predictive model that can precisely predict the correct class of the new data (unseen data) based on learning process that was applied on the training data. the Classification involves three steps which are: 
1. Construct the model by using Training set.
2. Evaluate the model by using Test set which determine the accuracy 
3. Based on the accuracy results acceptance, we will determine if the model can be used for prediction new data.
 
Our classification goal is to build a model that predicts our data set class label which is target which is either 1= targeted to have a heart disease or 0 = not targeted to have a heart disease based on the attributes.
We will partition the data into two partitions which are training and testing sets by trying three different sizes to determine the best size for our model using sample method we assign "True" value to replace attributes since our data set is small so we need a partitioning method that is suitable for small data size and construct samples with replacement.
For each partition size we construct three decision tree using different attribute selection method which are:Information gain (ID3), Gain ratio (C4.5) and Gini index (CART). For information gain we used ctree method to build the tree Where for gain ratio we used J48 method. For Gini index we used rpart method.For prediction we used predict method. Also, for evaluation and testing we used confusionMatrix method.

#### Classification packages:
We install and call predifined packages in R language to help us in classification process. which shown below:
```{r}
install.packages("partykit")
library(party)
library(partykit)
library(RWeka)
library(caret)
library(rpart)
library(rpart.plot)
```

#### Partitioning num.1 
We partition the data the data into (70% training, 30% testing). This result in 218 row in the training set and 84 row in the testing set.
```{r}
set.seed(1234)
ind=sample(2, nrow(dataset), replace=TRUE, prob=c(0.70 , 0.30))
train_data=dataset[ind==1,]
test_data=dataset[ind==2,]
dim(train_data)
dim(test_data)
```
##### Information Gain:
```{r}
myFormula<- target~ age + sex + cp + trestbps + chol + restecg + thalach + exang + oldpeak + slope + ca + thal
dataset_ctree<-ctree(myFormula, data=train_data)
table(predict(dataset_ctree), train_data$target)
```
The shown matrix based on training data.
###### Tree based on information gain:
```{r}
print(dataset_ctree)
plot(dataset_ctree)
plot(dataset_ctree,type="simple")
```
We applied two decision tree representations which are regular and simple tree.The result shows that Chest pain type (cp) had the highest information gain value so it was represented in first level (root). 

###### Prediction on test data and Confusion matrix:
```{r}
testPred<- predict(dataset_ctree,newdata=test_data)
results<- confusionMatrix(testPred,test_data$target,positive="1")
print(results)
```
The above matrix is based on applying our model on test data.
#### Schedule for classification Evaluation:
|Evaluation method|value|
|-----------------|-----|
|Accuracy|80.95%|
|Error Rate|19.05%|
|Sensitivity(Recall)|91.67%|
|Specificity|66.67%|
|Precision|78.57%|


##### Gain ratio:
```{r}
C45Fit <- J48(target~.,data=train_data)
table(predict(C45Fit), train_data$target)
```
The shown matrix based on training data.
###### Tree based on Gain ratio:
```{r}
print(C45Fit)
plot(C45Fit)
plot(C45Fit,type="simple")
```
We applied two decision tree representations which are regular and simple tree.The result shows A blood disorder (thal) had the highest gain ratio value so it was represented in first level (root). 

###### Prediction on test data and Confusion matrix:
```{r}
testPred<- predict(C45Fit,newdata=test_data)
results<- confusionMatrix(testPred,test_data$target,positive="1")
print(results)
```
The above matrix is based on applying our model on test data.
#### Schedule for classification Evaluation:
|Evaluation method|value|
|-----------------|-----|
|Accuracy|76.19%|
|Error Rate|23.81%|
|Sensitivity(Recall)|77.08%|
|Specificity|75%|
|Precision|80.43%|

##### Gini index
```{r}
fit.tree=rpart(target~., data=train_data, method="class",cp=0.008)
```

###### Tree based on Gini index :
```{r}
print(fit.tree)
rpart.plot(fit.tree)
```
We applied decision tree representation.The result shows that A blood disorder (thal) had the highest ΔGini (impurity reduction) value so it was represented in first level (root). 

###### Prediction on test data and Confusion matrix:
```{r}
testPred<- predict(fit.tree,newdata=test_data,type="class")
results<- confusionMatrix(testPred,test_data$target,positive="1")
print(results)
```
The above matrix is based on applying our model on test data.
#### Schedule for classification Evaluation:
|Evaluation method|value|
|-----------------|-----|
|Accuracy|76.19%|
|Error Rate|23.81%|
|Sensitivity(Recall)|75%|
|Specificity|77.78%|
|Precision|81.82%|

#### Partitioning num.2 
We partition the data the data into (75% training, 25% testing). This result in 233 row in the training set and 69 row in the testing set.
```{r}
set.seed(1234)
ind=sample (2, nrow(dataset), replace=TRUE, prob=c(0.75 , 0.25))
train_data=dataset[ind==1,]
test_data=dataset[ind==2,]
dim(train_data)
dim(test_data)
```
##### Information Gain:
```{r}
myFormula<- target~ age + sex + cp + trestbps + chol + restecg + thalach + exang + oldpeak + slope + ca + thal
dataset_ctree<-ctree(myFormula, data=train_data)
table(predict(dataset_ctree), train_data$target)
```
The shown matrix based on training data.
###### Tree based on information gain:
```{r}
print(dataset_ctree)
plot(dataset_ctree)
plot(dataset_ctree,type="simple")
```
We applied two decision tree representations which are regular and simple tree.The result shows Number of major vessels (ca) had the highest information gain value so it was represented in first level (root).

###### Prediction on test data and Confusion matrix:
```{r}
testPred<- predict(dataset_ctree,newdata=test_data)
results<- confusionMatrix(testPred,test_data$target,positive="1")
print(results)
```
The above matrix is based on applying our model on test data.
#### Schedule for classification Evaluation:
|Evaluation method|value|
|-----------------|-----|
|Accuracy|79.71%|
|Error Rate|20.29%|
|Sensitivity(Recall)|82.93%|
|Specificity|75%|
|Precision|82.93%|

##### Gain ratio:
```{r}
C45Fit <- J48(target~.,data=train_data)
table(predict(C45Fit), train_data$target)
```
The shown matrix based on training data.
###### Tree based on Gain ratio:
```{r}
C45Fit
plot(C45Fit)
plot(C45Fit,type="simple")
```

###### Prediction on test data and Confusion matrix:
```{r}
testPred<- predict(C45Fit,newdata=test_data)
results<- confusionMatrix(testPred,test_data$target,positive="1")
print(results)
```

#### Schedule for classification Evaluation:
|Evaluation method|value|
|-----------------|-----|
|Accuracy|79.71%|
|Error Rate|20.29%|
|Sensitivity(Recall)|85.37%|
|Specificity|71.43%|
|Precision|81.40%|

##### Gini index:
```{r}
fit.tree=rpart(target~., data=train_data, method="class",cp=0.008)
```
###### Tree based on Gini index :
```{r}
print(fit.tree)
rpart.plot(fit.tree)
```

###### Prediction on test data and Confusion matrix:
```{r}
testPred<- predict(fit.tree,newdata=test_data,type="class")
results<- confusionMatrix(testPred,test_data$target,positive="1")
print(results)
```
#### Schedule for classification Evaluation:
|Evaluation method|value|
|-----------------|-----|
|Accuracy|73.91%|
|Error Rate|26.09%|
|Sensitivity(Recall)|73.17%|
|Specificity|75%|
|Precision|81.08%|


#### Partitioning num.3 the data into 
We partition the data the data into (80% training, 20% testing). This result in 249 row in the training set and 53 row in the testing set.
```{r}
set.seed(1234)
ind=sample (2, nrow(dataset), replace=TRUE, prob=c(0.80 , 0.20))
train_data=dataset[ind==1,]
test_data=dataset[ind==2,]
dim(train_data)
dim(test_data)
```
##### Information Gain:
```{r}
myFormula<- target~ age + sex + cp + trestbps + chol + restecg + thalach + exang + oldpeak + slope + ca + thal
dataset_ctree<-ctree(myFormula, data=train_data)
table(predict(dataset_ctree), train_data$target)
```
The shown matrix based on training data.
###### Tree based on information gain:
```{r}
print(dataset_ctree)
plot(dataset_ctree)
plot(dataset_ctree,type="simple")
```

###### Prediction on test data and Confusion matrix:
```{r}
testPred<- predict(dataset_ctree,newdata=test_data)
results<- confusionMatrix(testPred,test_data$target,positive="1")
print(results)
```
#### Schedule for classification Evaluation:
|Evaluation method|value|
|-----------------|-----|
|Accuracy|83.02%|
|Error Rate|16.98%|
|Sensitivity(Recall)|89.66%|
|Specificity|75%|
|Precision|81.25%|

##### Gain ratio:
```{r}
C45Fit <- J48(target~.,data=train_data)
table(predict(C45Fit), train_data$target)
```
The shown matrix based on training data.
###### Tree based on Gain ratio:
```{r}
print(C45Fit)
plot(C45Fit)
plot(C45Fit,type="simple")
```

###### Prediction on test data and Confusion matrix:
```{r}
testPred<- predict(C45Fit,newdata=test_data)
results<- confusionMatrix(testPred,test_data$target,positive="1")
print(results)
```
#### Schedule for classification Evaluation:
|Evaluation method|value|
|-----------------|-----|
|Accuracy|79.25%|
|Error Rate|20.75%|
|Sensitivity(Recall)|82.76%|
|Specificity|75%|
|Precision|80%|

##### Gini index
```{r}
fit.tree=rpart(target~., data=train_data, method="class",cp=0.008)
```
###### Tree based on Gini index :
```{r}
print(fit.tree)
rpart.plot(fit.tree)
```

###### Prediction on test data and Confusion matrix:
```{r}
testPred<- predict(fit.tree,newdata=test_data,type="class")
results<- confusionMatrix(testPred,test_data$target,positive="1")
print(results)
```
#### Schedule for classification Evaluation:
|Evaluation method|value|
|-----------------|-----|
|Accuracy|75.47%|
|Error Rate|24.53%|
|Sensitivity(Recall)|75.86%|
|Specificity|75%|
|Precision|78.57%|


###Evaluation and Comparison of classification:
We applied several methods to evaluate classification models which are: accuracy, error rate, sensitivity (recall) , specificity, and precision. Accuracy means the ability of the model to classify the data correctly. It’s used to evaluate the performance of the model based on the test data. Error rate is the opposite of the accuracy. it means the ability of the model to classify the data incorrectly. Sensitivity means the ability of the model to correctly identify positive tuples. It also called Recall. Specificity is the opposite of the sensitivity. It means the ability of the model to correctly identify negative tuples. Finally, precision means the proportion of correctly predicted positive tuples out of all tuples predictive as positive.

##### Information Gain:
It measures the amount of information provided by a feature, information gain is used to determine the best feature which has the highest value to split the data at each node of the tree. It uses entropy which is a measure of impurity in a set of data. 

|  |70% training, 30% testing|75% training, 25% testing|80% training, 20% testing|
|--|-------------------------|-------------------------|-------------------------|
|Accuracy|80.95%|79.71%|83.02%|
|Error Rate|19.05%|20.29%|16.98%|
|Sensitivity(Recall)|91.67|82.93%|89.66%|
|Specificity|66.67%|75%|75%|
|Precision|78.57%|82.93%|81.25%|

##### Gain ratio:
It is a measure used in decision tree algorithms to select the best feature that has the highest value for splitting data. It takes into consideration both the information gain and the split information of a feature.

|  |70% training, 30% testing|75% training, 25% testing|80% training, 20% testing|
|-----------------|-----|-----|-----|
|Accuracy|76.19%|79.71%|79.25%|
|Error Rate|23.81%|20.29%|20.75%|
|Sensitivity(Recall)|77.08%|85.37%|82.76%|
|Specificity|75%|71.43%|75%|
|Precision|80.43%|81.40%|80%|

##### Gini index:
It is a measurement used in classifying the data and building decision trees, it is used to determine the splitting attribute. The gini index measure the impurity of the data and the attribute with the lowest gini index value will have the highest reduction impurity and will be chosen as the split attribute. 

|  |70% training, 30% testing|75% training, 25% testing|80% training, 20% testing|
|-----------------|-----|-----|-----|
|Accuracy|76.19%|73.91%|75.47%|
|Error Rate|23.81%|26.09%|24.53%|
|Sensitivity(Recall)|75%|73.17%|75.86%|
|Specificity|77.78%|75%|75%|


From the previous Schedule we notice that: 
- For information gain partitioning num.3 (80% training, 20% testing) had the highest accuracy value (83.02%).
- For Gain ratio partitioning num.2 (75% training, 25% testing) had the highest accuracy value (79.71%).
- For Gini index partitioning num.1 (70% training, 30% testing) had the highest accuracy value (76.19%).
Overall, information gain accuracy result was the highest so, we will choose partitioning num.3 (80% training, 20% testing)

#### The disicion tree information gain partitioning num.3 (80% training, 20% testing)
```{r}
set.seed(1234)
ind=sample (2, nrow(dataset), replace=TRUE, prob=c(0.80 , 0.20))
train_data=dataset[ind==1,]
test_data=dataset[ind==2,]
myFormula<- target~ age + sex + cp + trestbps + chol + restecg + thalach + exang + oldpeak + slope + ca + thal
dataset_ctree<-ctree(myFormula, data=train_data)
table(predict(dataset_ctree), train_data$target)
plot(dataset_ctree,type="simple")
```


### Clustering: 
Clustering is the task of arranging a set of objects in such a way that objects in the same group (cluster) are more comparable (in some sense) to those in other groups (clusters). It’s unsupervised learning which means it doesn’t have a class label. In this section we are going to partition our data using k-means.we are going to try three different k-means values which are (2,3 and 4).For each trial we will calculate the average silhouette ,total within-cluster sum of square and the BCubed (precision and recall).

About packages and methods, We use K-means methods to creat the clusters. Also, For plotting We use fviz_cluster. For evaluation we use fviz_nbclust and fviz_silhouette to calculate the silhouette value. We create a method to calculate Bcubes percision and recall.

#### Packages
```{r}
install.packages("factoextra")
library(factoextra)
library(NbClust)
library(cluster)
```


#### Removing the class label(target) before we partition our data 
We have to remove the class label(target) since the clustering is an unsupervised learning.

```{r}
dataBeforC<-dataset #in case we need the old data set(with the class label)
dataset <- dataset[, -which(names(dataset) == "target")]
```

since we applied discretization to the age we can't deal with factors during the clustering process Instead we are going to retrieve the old values before discretization.

```{r}
dataset$age <- AgeBeforeDis
```

#### Converting interger columns to numeric 

```{r}
dataset$sex <- as.numeric(dataset$sex ) 
dataset$cp <- as.numeric(dataset$cp )
dataset$trestbps <- as.numeric(dataset$trestbps  ) 
dataset$restecg <- as.numeric(dataset$restecg ) 
dataset$thalach <- as.numeric(dataset$thalach)
dataset$exang <- as.numeric(dataset$exang) 
dataset$slope <- as.numeric(dataset$slope)
dataset$ca <- as.numeric(dataset$ca) 
dataset$thal <- as.numeric(dataset$thal)
dataset$age <- as.numeric(dataset$age)
```

here is a simple representation to our structure after converting all data into numeric

```{r}
str(dataset)
```


####Optimal number of clusters
```{r}
fviz_nbclust(dataset, kmeans, method = "silhouette")+labs(subtitle ="Silhouette method")
```

In this graph, the number of clusters is set to 2. The highest point on the graph represents the optimal number of clusters. This corresponds to the number of clusters that yield the highest average silhouette score, indicating the best clustering solution.

It's worth noting that the silhouette score is just one method used to estimate the optimal number of clusters. Other approaches and metrics can also be employed for this purpose.

#### calculate k-mean k=2
```{r}
km <- kmeans(dataset, 2, iter.max = 140 , algorithm="Lloyd", nstart=100) 
km
```

#### plot k-mean
```{r}
fviz_cluster(list(data = dataset, cluster = km$cluster),        
             ellipse.type = "norm", geom = "point", stand = FALSE,          
             palette = "jco", ggtheme = theme_classic())
```
From the above results and the cluster representation we notice that there's two clusters. First cluster contains 183 objects with total sum of square 5687.175. Second cluster contains 119 object with total sum of square 3524.142. The value of the total sum of square indicate that the larger value means the larger seperation so,the object in the first cluster are more separated than the object in the second cluster. Also, we notice that there's few outliers in each cluster. 
#### Average silhouette
```{r}
sil <- silhouette(km$cluster, dist(dataset)) 
rownames(sil) <- rownames(dataset)
```

```{r}
fviz_silhouette(sil)
```
The average silhouette width of 0.53 suggests that the data points in a clustering analysis are well-matched to their assigned clusters and the clusters themselves are reasonably separated from each other. A positive average silhouette width indicates that, on average, the data points are well-clustered and have a higher similarity to their own cluster compared to other clusters.

#### Total within-cluster-sum of square
```{r}
km$tot.withinss
```
####BCubed precision and recall
```{r}
cluster_assignments <- c(km$cluster)
ground_truth_labels <- c(dataBeforC$target)

data <- data.frame(cluster = cluster_assignments, label = ground_truth_labels)

# Function to calculate BCubed precision and recall
calculate_bcubed_metrics <- function(data) {
  n <- nrow(data)
  precision_sum <- 0
  recall_sum <- 0

  for (i in 1:n) {
    cluster <- data$cluster[i]
    label <- data$label[i]
    
# Count the number of items from the same category within the same cluster
same_category_same_cluster <- sum(data$label[data$cluster == cluster] == label)
    
# Count the total number of items in the same cluster
total_same_cluster <- sum(data$cluster == cluster)
    
# Count the total number of items with the same category
total_same_category <- sum(data$label == label)
    
# Calculate precision and recall for the current item and add them to the sums
precision_sum <- precision_sum + same_category_same_cluster /total_same_cluster
recall_sum <- recall_sum + same_category_same_cluster / total_same_category
  }

  # Calculate average precision and recall
  precision <- precision_sum / n
  recall <- recall_sum / n

  return(list(precision = precision, recall = recall))
}

# Calculate BCubed precision and recall
metrics <- calculate_bcubed_metrics(data)

# Extract precision and recall from the metrics
precision <- metrics$precision
recall <- metrics$recall

# Print the results
cat("BCubed Precision:", precision, "\n")
cat("BCubed Recall:", recall, "\n")
```

The Bcubed precision value indicates the percentage of correctly grouped data points in each cluster, with an average of approximately 53.47%. A higher precision value suggests a better ability of the clustering algorithm to assign similar data points to the same cluster while avoiding unrelated ones. On the other hand, the Bcubed recall value represents the percentage of relevant data points captured by the algorithm in each cluster, averaging around 55.23%. Higher recall values indicate the algorithm's ability to include all the relevant data points in each cluster.
In summary, the precision value is slightly lower than the recall value, indicating that the clustering algorithm may include some unrelated data points in a cluster, resulting in a slightly lower precision. However, it's worth noting that the difference between precision and recall is relatively small.

#### calculate k-mean k=3
```{r}
km <- kmeans(dataset, 3, iter.max = 140 , algorithm="Lloyd", nstart=100) 
km
```

#### plot k-mean
```{r}
fviz_cluster(list(data = dataset, cluster = km$cluster),            
             ellipse.type = "norm", geom = "point", stand = FALSE,     
             palette = "jco", ggtheme = theme_classic())
```
From the above results and the cluster representation we notice that there's three clusters. First cluster contains 136 objects with total sum of square 1917.605. Second cluster contains 87 objects with total sum of square 1762.273. The third cluster contains 79 objects which is the smallest with total sum of square 1379.237.The value of the total sum of square indicate that the larger value means the larger separation so,the object in the first cluster are more separated than the objects in the other clusters and the high number of objects may have impacted this value since WSS is influenced by the number of observations.smallest WSS was for the third cluster that indicates that the objects are more compact.Also, we notice that there's few outliers in the third cluster and only one outlier in the second cluster.

#### Average silhouette
```{r}
sil <- silhouette(km$cluster, dist(dataset))
rownames(sil) <- rownames(dataset)
```

```{r}
fviz_silhouette(sil)
```
The average silhouette width is 0.47. This value indicates that, on average, the data points within the clusters are reasonably well-separated from each other and exhibit some similarity to the points in their neighboring clusters. However, it's important to note that the interpretation of the silhouette width heavily depends on the specific data set and problem domain. Different data set may have different standards for what constitutes a good or bad silhouette width.

#### Total within-cluster-sum of square
```{r}
km$tot.withinss
```

#### BCubed precision and recall
```{r}
cluster_assignments <- c(km$cluster)
ground_truth_labels <- c(dataBeforC$target)

data <- data.frame(cluster = cluster_assignments, label = ground_truth_labels)

# Function to calculate BCubed precision and recall
calculate_bcubed_metrics <- function(data) {
  n <- nrow(data)
  precision_sum <- 0
  recall_sum <- 0

  for (i in 1:n) {
    cluster <- data$cluster[i]
    label <- data$label[i]
    
# Count the number of items from the same category within the same cluster
same_category_same_cluster <- sum(data$label[data$cluster == cluster] == label)
    
# Count the total number of items in the same cluster
total_same_cluster <- sum(data$cluster == cluster)
    
# Count the total number of items with the same category
total_same_category <- sum(data$label == label)
    
# Calculate precision and recall for the current item and add them to the sums
precision_sum <- precision_sum + same_category_same_cluster /total_same_cluster
recall_sum <- recall_sum + same_category_same_cluster / total_same_category
  }

  # Calculate average precision and recall
  precision <- precision_sum / n
  recall <- recall_sum / n

  return(list(precision = precision, recall = recall))
}

# Calculate BCubed precision and recall
metrics <- calculate_bcubed_metrics(data)

# Extract precision and recall from the metrics
precision <- metrics$precision
recall <- metrics$recall

# Print the results
cat("BCubed Precision:", precision, "\n")
cat("BCubed Recall:", recall, "\n")
```
The Bcubed precision value of 0.5250096 suggests that approximately 52.5% of the items within each cluster are correctly assigned to the same cluster as other similar items. This indicates that the clustering algorithm has some success in grouping similar items together.  The Bcubed recall value of 0.3668491 indicates that around 36.7% of the items that should belong to the same cluster are indeed assigned to that cluster. This means that the algorithm is capturing only a fraction of the items that should be grouped together.

#### calculate k-mean k=4
```{r}
km <- kmeans(dataset, 4, iter.max = 140 , algorithm="Lloyd", nstart=100) 
km
```

#### plot k-mean
```{r}
fviz_cluster(list(data = dataset, cluster = km$cluster),            
             ellipse.type = "norm", geom = "point", stand = FALSE,         
             palette = "jco", ggtheme = theme_classic()) 
```
From the above results and the cluster representation we notice that there's four clusters. First cluster contains 60 objects with total sum of square 930.4175. Second cluster contains 73 objects with total sum of square 677.2899. The third cluster contains 99 objects  with total sum of square 907.0856. The fourth cluster contains 70 objects  with total sum of square 1140.7632. The value of the total sum of square indicate that the larger value means the larger separation so,the object in the fourth cluster are more separated than the objects in the other clusters .smallest WSS was for the second cluster that indicates that the objects are more compact. Also, we notice that there's few outliers in the first and the fourth clusters. 

#### Average silhouette
```{r}
sil <- silhouette(km$cluster, dist(dataset))
rownames(sil) <- rownames(dataset)
```

```{r}
fviz_silhouette(sil)
```
the average silhouette width is 0.4, it suggests that, on average, the data points are reasonably well-clustered. This means that the data points within each cluster have a moderate level of separation from data points in other clusters.
#### Total within-cluster-sum of square
```{r}
km$tot.withinss
```
The closer the points in a cluster are to the centroid, the better. A smaller distance indicates that the points are closer to the centroid, which is desirable. On the other hand, a larger distance signifies that the points are farther away from the centroid, which is less desirable.

Hence, the best choice for the K-means algorithm is the one that minimizes the distance between the points and the centroid, resulting in compact and well-separated clusters.
In this case the best k-means is 3655.556
#### BCubed precision and recall
```{r}
cluster_assignments <- c(km$cluster)
ground_truth_labels <- c(dataBeforC$target)

data <- data.frame(cluster = cluster_assignments, label = ground_truth_labels)

#Function to calculate BCubed precision and recall
calculate_bcubed_metrics <- function(data) {
  n <- nrow(data)
  precision_sum <- 0
  recall_sum <- 0

  for (i in 1:n) {
    cluster <- data$cluster[i]
    label <- data$label[i]
    
#Count the number of items from the same category within the same cluster
same_category_same_cluster <- sum(data$label[data$cluster == cluster] == label)
    
#Count the total number of items in the same cluster
total_same_cluster <- sum(data$cluster == cluster)
    
#Count the total number of items with the same category
total_same_category <- sum(data$label == label)
    
#Calculate precision and recall for the current item and add them to the sums
precision_sum <- precision_sum + same_category_same_cluster /total_same_cluster
recall_sum <- recall_sum + same_category_same_cluster / total_same_category
  }

  #Calculate average precision and recall
  precision <- precision_sum / n
  recall <- recall_sum / n

  return(list(precision = precision, recall = recall))
}

#Calculate BCubed precision and recall
metrics <- calculate_bcubed_metrics(data)

# Extract precision and recall from the metrics
precision <- metrics$precision
recall <- metrics$recall

#Print the results
cat("BCubed Precision:", precision, "\n")
cat("BCubed Recall:", recall, "\n")
```
A precision of 0.5502816 suggests that approximately 55.0% of the items within the clusters are correctly assigned to the same cluster as other similar items. This indicates a moderate level of precision in the clustering results.  A recall of 0.2854758 suggests that approximately 28.5% of the items that should be grouped together are correctly assigned to the same cluster. This indicates that the clustering algorithm captures only a fraction of the items that should belong to the same clusters


|                                          | k=2(Best)                        | k=3                                     | k=4                                                   |
|------------------------------------------|----------------------------------|-----------------------------------------|-------------------------------------------------------|
| Average Silhouette width for each cluter | cluter1=0.53,cluter2=0.53       | cluter1=0.46,cluter2=0.53,cluter3=0.43 | cluter1=0.39,cluter2=0.39,cluter3=0.47 ,cluter4=0.37 |
| Average Silhouette width for all cluters | 0.53                             | 0.47                                    | 0.4                                                   |
| Total within-cluster sum of square       | 9211.418                        |   5059.114                                   |  3655.556                                                     |
| BCubed (precision)                       | 0.5347233                        | 0.5250096                               | 0.5502816                                             |
| BCubed (recall)                          | 0.5523007                        | 0.3668491                               | 0.2854758                                             |
| Visualization                            | all of the figures is shown above | all of the figures is shown above       |    all of the figures is shown above               |

### Findings:
In classification, We perform multiple decision trees using different partitioning splits and different attribute selection method we chose that had the best accuracy which is partitioning data into 80% training and 20% testing with information gain attribute selection. It had the highest accuracy (83.02%) indicating that the model had the best performance in predicting the correct class labels(targeted or not targeted to heart disease). In the other hand, the error rate was low (16.98%) since it equals (1-accuracy) and it shows the percent of incorrect predictions. For sensitivity it was (89.66%) which is high and tells that the model can well predict the positive instances (targeted to heart attack) and for specificity it was (75%)  which is lower than the sensitivity and it tells how the model can predict the negative instances (not targeted to heart attack). Lastly, the precision was (81.25%) indicating  the true positive instances over all that instances the model predicted as positive.

The decision tree had 6 pathes from the root to the leaves so, we will have 6 rules.
Rules:
1- if number of major vessels (ca) <= 0 and exercise induced angina (exang) <= 0 and  blood disorder (thal) <= 2 then target to have heart disease (target= 1) 
2- if number of major vessels (ca) <= 0 and exercise induced angina (exang) <= 0 and  blood disorder (thal) > 2 then target to have heart disease (target= 1) 
3- if number of major vessels (ca) <= 0 and exercise induced angina (exang) > 0 and chest pain type (cp) <= 0 then not target to have heart disease (target= 0) 
4- if number of major vessels (ca) <= 0 and exercise induced angina (exang) > 0 and chest pain type (cp) > 0 then  target to have heart disease (target= 1) 
5- if number of major vessels (ca) > 0 and  chest pain type (cp) <= 0 then not target to have heart disease (target= 0)
6- if number of major vessels (ca) > 0 and  chest pain type (cp) > 0 then target to have heart disease (target= 1).
The decision tree illustrates that 4 attributes were used in predicting the class label which are (number of major vessels (ca), exercise induced angina (exang),blood disorder (thal) and chest pain type (cp)) where the other attributes had no effect in the class label prediction.


In Clustering, As previously mentioned, we used the K-means approach with three different values of K (K=2,3,4) to determine the optimal number of clusters. The average silhouette width, total within-cluster sum of squares, precision, and recall were used to evaluate the results.  We came at the following findings after studying the data:

The average silhouette width for K=2 was 0.53, showing that objects within a cluster were closely grouped and distinguishable from objects in other clusters. The overall within-cluster sum of squares was 9211.418, identifying that the variation within the clusters was rather substantial.Precision and recall were 0.5347233 and 0.5523007, respectively.This demonstrates a balanced capability to accurately capture instances within clusters and recognize relevant points.

The average silhouette width dropped to 0.47 for K=3. The overall within-cluster sum of squares decreased to 5059.114, indicating that there is less diversity within the clusters than when K=2 was used. The accuracy and recall scores were 0.5250096 and 0.3668491, indicating a significantly reduced capacity to collect occurrences and identify all relevant points.

The average silhouette width dropped to 0.4 for K=4. The variance within the clusters is even lower than when K=3, as evidenced by the total within-cluster sum of squares dropping to 3655.556. With precision and recall values of 0.5502816 and 0.2854758, respectively, there was a greater capacity to precisely record instances, but a reduced capacity to recognize every pertinent point.

The conclusion of these findings lead us to that the 2-Mean clustering model to be optimal. It had the highest average silhouette width, indicating tight and well-separated clusters. Additionally, it exhibited the least overlap between clusters compared to the other models. Although it had higher variance within the clusters, the precision and recall values were relatively balanced, indicating a good ability to accurately capture instances and identify relevant points.


Both approaches (Classification and Clustering)were helpful for building models for our data set that can help in reaching our goal which is protecting people from heart attacks, but since our data set  includes class label which is target, classification was considered the best option to predict the possibilities of having heart attack based on the attributes also the model had great accuracy.

