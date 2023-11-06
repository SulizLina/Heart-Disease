#Heart diseases

### Our goal:

The goal of collecting a dataset on heart diseases is to gather relevant information about the factors that influence the risk of gaining heart diseases. The results of analyzing this dataset will be utilized in various beneficial ways, research advancements, researchers work towards improving the usnderstanding and management of heart disease, and increasing awareness among the public regarding the common factors that can contribute to heart diseases.

### Classification goal:

The aim is to predict whether a person is at risk of having a certain type or level of heart disease based on their vital signals, such as age, sex, blood pressure, and cholesterol levels. The objective is to build a predictive model that accurately categorizes individuals into two groups: those likely to have a future heart disease (labeled as 1) and those unlikely to have a future heart disease (labeled as 0).

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

###Importing libraries
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
###Reading the data:
```{r}
data= read.csv('DataSet/heart.csv')
head(data)
```
###Check the duplication:
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

##Number of rows and coloumns:
To represent our new data set columns and rows we use this code. It shows that our new data set consists of 14 coloumns and 302 rows.
```{r}
ncol(dataset)
nrow(dataset)
```

##Statistical measures:
We will represent the five number summary, the mean and the variance for the numerical attributes. These values will give us an overview about our attributes. Also, it could be useful for smoothing and handling the outliers, missing and wrong values.

####For the Age attribute:
```{r}
summary(dataset$age)
var(dataset$age)
```
####For the resting blood pressure (trestbps) attribute:
```{r}
summary(dataset$trestbps)
var(dataset$trestbps)
```
####For the serum cholestoral (chol) attribute:
```{r}
summary (dataset$chol)
var(dataset$chol)
```
####For the maximum heart rate achieved (thalach) attribute:
```{r}
summary(dataset$thalach)
var(dataset$thalach)
```
####for the ST depression (oldpeak) attribute:
```{r}
summary(dataset$oldpeak)
var(dataset$oldpeak)
```

###Boxplot:
####For the Age attribute:
```{r}
boxplot(dataset$age)
```
The age summary and boxplot show that all people in our data set have ages ranging from 29 to 77.There are no outliers in age attribute. The median (55.50) is close to the 3rd quartile (61.00) indicating a slightly skewed distribution.

####For the resting blood pressure (trestbps) attribute:
```{r}
boxplot(dataset$trestbps)
```
The trestbps summary and boxplot show that resting blood pressure values (trestbps) span from 94mmHg to 200mmHg. There are few outliers, that we will deal with in preprocessing section. The distribution of the data appears to be symmetric where the median (130.0mmHg) line is centered within the box, and the whiskers are of similar length on both sides.

####For the serum cholestoral (chol) attribute:
```{r}
boxplot(dataset$chol)
```
The serum cholesterol (chol) summary and boxplot shows that the attribute range from 126 to 564. We notice from the graph that there are few outliers where all of them are close to each other except one of them. As we mention before we will handle the outliers in preprocessing section.

####For the maximum heart rate achieved (thalach) attribute:
```{r}
boxplot(dataset$thalach)
```
The summary and boxplot of maximum heart rate (thalach) data reveal that the maximum heart rate values range from 71.0 to 202.0. The boxplot visually presents the distribution of the attribute data appears to be slightly skewed distribution.

####For the ST depression (oldpeak) attribute:
```{r}
boxplot(dataset$oldpeak)
```
The summary and boxplot of ST depression (oldpeak) attribute range from 0 to 6.2. Also, it displays only the box and the upper whisker and it does not have a lower whisker. That means that there are no values below the lower fence.

###Graphical representations:
####Pie chart for our class label (target).
```{r}
tab <- dataset$target %>% table()
percentages <- tab %>% prop.table() %>% round(3) * 100
txt <- paste0(names(tab), '\n', percentages, '%')
pie(tab, labels = txt, main="percentage of the target")
```
The pie chart help us to know the percentage of the people in our data set who might be targeted by heart diseases. More than half of the people (represented by 54.3%) has high potentials of getting infected. In other side less than half of the people (represented by 45.7%) has low potentials of getting infected.

####Graph between the sex and the target.
```{r}
ggplot(dataset, aes(x = sex, fill = as.factor(target))) +
  geom_bar(stat = "count", position = "stack", width = 0.6, show.legend = TRUE) +
  geom_text(aes(label = after_stat(count)), stat = 'count', position = position_stack(vjust = 0.5)) +
  labs(x = "Sex", y = "Count",fill="target") 
```
The chart represent sex attribute and our class label (target). Most of the people in our data set were male. For male, the target is almost equally distributed. For female, the majority of them are considered targeted. 

####Graph between the chest pain type (cp) and the target.
```{r}
ggplot(dataset, aes(x = cp, fill = as.factor(target))) +
  geom_bar(stat = "count", position = "stack", width = 0.6, show.legend = TRUE) +
  geom_text(aes(label = after_stat(count)), stat = 'count', position = position_stack(vjust = 0.5)) +
  labs(x = "cp", y = "Count",fill="target") 
```
The chart represent chest pain type (cp) attribute and our class label (target).The chart shows that "0" which indicate asymptomatic type are people who don't have high potentiality to be infected. In contrast, The other three types ("1" atypical angina ,"2" non-anginal pain ,"3" typical angina) have higher potential to be infected.

####Graph between the number of major vessels (ca) and the target.
```{r}
ggplot(dataset, aes(x = ca, fill = as.factor(target))) +
  geom_bar(stat = "count", position = "stack", width = 0.6, show.legend = TRUE) +
  geom_text(aes(label = after_stat(count)), stat = 'count', position = position_stack(vjust = 0.5)) +
  labs(x = "ca", y = "Count",fill="target")
```
The chart represent the number of major vessels colored by flourosopy (ca) attribute and our class label (target). The graph shows that ca values are correlated to the possibility of getting infected. Where for vessel "0" approximately three quarters of people are infected.In contrast "1,2,3" vessels show that the targeted people have less percentage of getting infected. Also, we detected a wrong value which is 4 but, we will handle it in the preprocessing section.

####Graph between maximun heart rate (thalach) and the target.
```{r}
plot(dataset$target,dataset$thalach)
```
The scatter plot represent maximum heart rate achieved (thalach) attribute and our class label (target). We notice from the graph that the majority of targeted people has high thalach value (thalach>130). 

###Scatter Plot for Age and resting blood sugar (trestbps).
```{r}
plot(dataset$age, dataset$trestbps, pch = 16, col = "black", xlab = "Age", ylab = "trestbps", main = "Scatter Plot for Age and trestbps")
abline(lm(dataset$trestbps ~ dataset$age), col = "red", lwd = 2)
```
The scatter plot represent resting blood pressure (trestbps) and age attributes. The line shows that there is a slight positive correlation between them. Also, we noticed that trestbps has high values between 50 to 60 years.

###Scatter Plot for Maximum heart rate (thalach) and ST depression induced by exercise (oldpeak).
```{r}
plot(dataset$thalach, dataset$oldpeak, pch = 16, col = "black", xlab = "thalach", ylab = "oldpeak", main = "Scatter Plot for thalach and oldpeak")
abline(lm(dataset$oldpeak ~ dataset$thalach), col = "red", lwd = 2)
```
The scatter plot represent ST depression induced by exercise relative to rest (oldpeak) and a blood disorder called thalassemia Value (thalach) attributes. The line shows that there is a negative correlation between them. Also, we noticed that the most frequent value for oldpeak is zero which was centrelized between 150 to 185.
```{r}
plot(dataset$chol,dataset$target)
```
The scatter plot represent serum cholestoral (chol) and our class label (target) attributes. We think that there is a very weak correlation between them since, the representation of 0 and 1 in y axis are almost similar. We will prove that when we apply the correlation coefficient in the preprocessing section.

####Graph between the fasting blood sugar(fbs) and the target.
```{r}
ggplot(dataset, aes(x = fbs, fill = as.factor(target))) +
  geom_bar(stat = "count", position = "stack", width = 0.6, show.legend = TRUE) +
  geom_text(aes(label = after_stat(count)), stat = 'count', position = position_stack(vjust = 0.5)) +
  labs(x = "fbs", y = "Count",fill="target")
```
The chart represent the fasting blood sugar(fbs) (1 means grater than 120 and 0 means less than 120) attribute and our class label (target). From the graph, Either fbs is less or greater than 120 they both have the same percentage of getting infected which is approximately 50% so, we determine that they are not correlated. We will prove that when we apply the correlation coefficient in the preprocessing section.

##Preprocessing:


###Detecting the missing values: 
```{r}
sum(is.na(data))
```
since the output is 0 so, there is no missing values in our data set.

###Detecting outliers:

####for the Age attributes:
```{r}
boxplot.stats(dataset$age)$out
```
There is no outliers in the age attribute.

####for the resting blood pressure (trestbps) attributes:
```{r}
boxplot.stats(dataset$trestbps)$out
```
From the previous code we notice that we have 9 outliers but, the pressure can be over 180 in some people which considered high level of pressure so we will not consider it as an outlier.

####for the serum cholestoral (chol) attributes:
```{r}
boxplot.stats(dataset$chol)$out
```
From the previous code we notice that we have 5 outliers but, 4 of them were close to the rest of the serum cholestoral values except 564 so we will handle it by smoothing using mean of the value in the following steps. 

####for the maximum heart rate achieved (thalach) attributes:
```{r}
boxplot.stats(dataset$thalach)$out
```
From the previous code we notice that we have one outlier but, we will not consider it as an outlier since it's medically acceptable for the maximum heart rate achieved.

####for the ST depression (oldpeak) attributes:
```{r}
boxplot.stats(dataset$oldpeak)$out
```
From the previous code we notice that we have 5 outliers but, we will not consider them as outliers since these values are medically acceptable for the ST depression.

###Handling outliers values:
We have one outlier that we will handle which is in serum cholestoral (chol) attributes with the value 564. Since it's only one value we're going to handle it manually by replacing it with the mean. We extracted the mean value from the summary code.
```{r}
dataset[129,5]=246.5
```

###Detecting wrong values:
We are going to check if there is any wrong values out of the attributes values range.

####The number of major vessels (ca) attribute:
```{r}
dataset[dataset$ca != "3" & dataset$ca != "1" & dataset$ca != "2" & dataset$ca != "0",]
```
We noticed that value number 4 is considered as a wrong value. It has been in 4 rows. We will handle it in the following step.

####Sex attribute:
```{r}
dataset[dataset$sex != "0" & dataset$sex!= "1",]
```
There's no wrong value in the sex attribute.

####Fasting blood sugar (fbs) attribute:
```{r}
dataset[dataset$fbs != "0" & dataset$fbs != "1",]
```
There's no wrong value in the fasting blood sugar attribute.

####Resting electrocardiographic results (restecg) attribute:
```{r}
dataset[dataset$restecg != "0" & dataset$restecg != "1" & dataset$restecg != "2" ,]
```
There's no wrong value in the Resting electrocardiographic results attribute.

####Exercise induced angina (exang) attribute:
```{r}
dataset[dataset$exang != "0" & dataset$exang != "1",]
```
There's no wrong value in the Exercise induced angina attribute.

####The slope of the peak exercise ST segment (slope) attribute:
```{r}
dataset[dataset$slope != "0" & dataset$slope != "1" & dataset$slope != "2" ,]
```
There's no wrong value in the slope of the peak exercise ST segment attribute.

####A blood disorder (thal) attribute:
```{r}
dataset[dataset$thal != "1" & dataset$thal != "2" & dataset$thal != "3" ,]
```
We noticed that value number 0 is considered as a wrong value. It has been in 2 rows. We will handle it in the following step.

####Result for detecting wrong values: 
We notice that number of major vessels and blood disorder attributes has wrong values (out of range).These values will be handled by using central tendencies values.

###Handling wrong values:
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

###Discretization: 
We will use discretization for resting blood pressure (trestbps) and age attributes which help us to form intervals and each interval has categorical label.

####Resting blood pressure (trestbps):
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
####Age:
```{r}
dataset$age <- ifelse(dataset$age <= 16, "Children",
                      ifelse(dataset$age <= 39, "Young Adults",
                                 ifelse(dataset$age <= 59, "Middle-aged Adults",
                                        ifelse(dataset$age < 99, "Old Adults",0))))
agedis <- dataset$age
print(agedis)
```

###Normalization: 
The normalization step will transform the attributes values into smaller range which will help us to provide an equal weight for the attributes.
We will use min-max normalization for many attributes.

####Call min-max normalize function
```{r}
min_max_normalize <- function(x) {
  (x - min(x)) / (max(x) - min(x))}
```

####Applying normlization function in serum cholestoral (chol):
```{r}
dataset$chol <- min_max_normalize(dataset$chol)
print(dataset$chol)
```
####Applying normlization function in ST depression induced by exercise (oldpeak):
```{r}
dataset$oldpeak <- min_max_normalize(dataset$oldpeak)
print(dataset$oldpeak)
```

####Applying normlization function in maximum heart rate achieved (thalach):
```{r}
dataset$thalach <- min_max_normalize(dataset$thalach)
print(dataset$thalach)
```


###Encoding:
Most of our attributes values are already encoded in the data set. We will encode the age and resting blood pressure (trestbps).

For example the sex is encoded to 0's and 1's where 0= female and 1= male as a binary type. What's shown below are the values of sex column.
```{r}
sexencoding <- dataset$sex
print(sexencoding)
```
####Encoding for resting blood pressure (trestbps):
```{r}
dataset$trestbps=factor(dataset$trestbps,levels=c("Low","Normal","Elevated","High Stage 1","High Stage 2","High Stage 3"),labels=c(0,1,2,3,4,5))
```

####Encoding for age:
```{r}
dataset$age=factor(dataset$age,levels=c("Children","Young Adults","Middle-aged Adults","Old Adults"),labels=c(0,1,2,3))
```


##Correlation Analysis:
We will find the correlation between each attributes and the class label (target). For the nominal data we will use chi-square and for the numeric data we will use correlation coefficient. This will help us to determine the most important and correlated attributes to the target.

###Chi-square for nominal data:

####Sex:
```{r}
csex=chisq.test(dataset$sex , dataset$target)
print(csex)
```
####Chest pain type (cp):
```{r}
ccp=chisq.test(dataset$cp , dataset$target)
print(ccp)
```
####Fasting blood sugar (fbs):
```{r}
cfbs=chisq.test(dataset$fbs , dataset$target)
print(cfbs)
```
####Resting electrocardiographic result (restecg):
```{r}
crestecg=chisq.test(dataset$restecg , dataset$target)
print(crestecg)
```
####Exercise induced anginal (exang):
```{r}
cexang=chisq.test(dataset$exang , dataset$target)
print(cexang)
```
####The slope of the peak exercise ST segment (slope):
```{r}
cslope=chisq.test(dataset$slope , dataset$target)
print(cslope)
```
####Number of major vessels (ca):
```{r}
cca=chisq.test(dataset$ca , dataset$target)
print(cca)
```
####A blood disorder (thal):
```{r}
cthal=chisq.test(dataset$thal , dataset$target)
print(cthal)
```
####Age:
The age attribute after discretization.
```{r}
cage= chisq.test(dataset$age, dataset$target)
print(cage)
```
####Resting blood pressure (trestbps):
The trestbps attribute after discretization.
```{r}
ctrestbps=chisq.test(dataset$trestbps ,dataset$target)
print(ctrestbps)
```
###Chi-square Results:
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


###Correlation coefficient for numeric data:

####Serum cholestoral (chol):
```{r}
cchol=cor(dataset$chol ,dataset$target)
print(cchol)
```
####Maximum heart rate achieved (thalach):
```{r}
cthalach=cor(dataset$thalach ,dataset$target)
print(cthalach)
```
####ST depression indicated by exercise relative to rest (oldpeak):
```{r}
coldpeak=cor(dataset$oldpeak ,dataset$target)
print(coldpeak)
```
###Correlation coefficient Results:
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


###Classification:

```{r}
library(party)
```

####Partitioning num.1 the data into (70% training, 30% testing)
```{r}
set.seed(1234)
ind=sample (2, nrow(dataset), replace=TRUE, prob=c(0.70 , 0.30))
train_data=dataset[ind==1,]
test_data=dataset[ind==2,]
```
#####Information Gain:
```{r}

```
#####Gain ratio:
```{r}

```
#####Gini index
```{r}
```

####Partitioning num.2 the data into (75% training, 25% testing)
```{r}
set.seed(1234)
ind=sample (2, nrow(dataset), replace=TRUE, prob=c(0.75 , 0.25))
trainData=dataset[ind==1,]
testData=dataset[ind==2,]
```
#####Information Gain:
```{r}
myFormula<- target~Age+sex+cp+trestbps+chol+fbs+restecg+thalach+exang+oldpeak+slope+ca+thal
dataset_ctree<-ctree(myFormula, data=trainData)
table(predict(dataset_ctree), trainData$target)
```
#####Gain ratio:
```{r}

```
#####Gini index
```{r}
```

####Partitioning num.3 the data into (80% training, 20% testing)
```{r}
set.seed(1234)
ind=sample (2, nrow(dataset), replace=TRUE, prob=c(0.80 , 0.20))
train_data=dataset[ind==1,]
test_data=dataset[ind==2,]
```
#####Information Gain
```{r}

```
#####Gain ratio:
```{r}

```
#####Gini index
```{r}
```


```{r}
dim(trainData)
dim(testData)
head(testData)
```


