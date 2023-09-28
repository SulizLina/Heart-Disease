# heart diseases

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
|cp|chest pain type|Ordinal|from 0 to 3 which 0=asymptomatic , 1= atypical angina , 2= non-anginal pain , 3= typical angina|
|trestbps|resting blood pressure (in mm Hg on admission to the hospital)|Numeric|from 94 to 200|
|chol|serum cholestoral in mg/dl| Numeric| from 126 to 564|
|fbs|fasting blood sugar greater than 120 mg/dl)|Binary| 1=true , 0=false|
|restecg|resting electrocardiographic results|Ordinal|from 0 to 2 which 0=showing probable or definite left ventricular hypertrophy by Estes’ criteria , 1=normal , 2=having ST-T wave abnormality|
|thalach|maximum heart rate achieved|Numeric| from 71 to 202|
|exang|exercise induced angina|Binary|1=yes , 0=no|
|oldpeak|ST depression induced by exercise relative to rest|Numeric|from 0 to 6.2|
|slope|the slope of the peak exercise ST segment|Ordinal| from 0 to 2 which 0=downsloping , 1=flat , 2=upsloping|
|ca|The number of major vessels(0-3)colored by flourosopy| Numeric| from 0 to 3|
|thal|A blood disorder called thalassemia Value|Ordinal| from 0 to 2 which 0=normal , 1= fixed defect , 2=reversible defect|

```{r}
library(ggplot2)
```


```{r}
dataset= read.csv('heart.csv')
head(dataset)
```
###Missing values: 
```{r}
sum(is.na(dataset))
```
since the output is 0 so, there is no missing values in our datsset.

###statistical measures:
we will present the five number summary, the mean and the variance for the numerical attributes.

####for the Age attributes:
```{r}
summary(dataset$age)
var(dataset$age)
```
####for the resting blood pressure (trestbps) attributes:
```{r}
summary(dataset$trestbps)
var(dataset$trestbps)
```
####for the serum cholestoral (chol) attributes:
```{r}
summary (dataset$chol)
var(dataset$chol)
```
####for the maximum heart rate achieveddl (thalach) attributes:
```{r}
summary (dataset$thalach)
var(dataset$thalach)
```
####for the ST depression (oldpeak) attributes:
```{r}
summary(dataset$oldpeak)
var(dataset$oldpeak)
```
####for the number of major vessels (ca) attributes:
```{r}
summary(dataset$ca)
var(dataset$ca)
```
###Outliers:
####for the Age attributes:
```{r}
boxplot.stats(dataset$age)$out
```
####for the resting blood pressure (trestbps) attributes:
```{r}
boxplot.stats(dataset$trestbps)$out
```
####for the serum cholestoral (chol) attributes:
```{r}
boxplot.stats(dataset$chol)$out
```
####for the maximum heart rate achieveddl (thalach) attributes:
```{r}
boxplot.stats(dataset$thalach)$out
```
####for the ST depression (oldpeak) attributes:
```{r}
boxplot.stats(dataset$oldpeak)$out
```
####for the number of major vessels (ca) attributes:
```{r}
boxplot.stats(dataset$ca)$out
```
###Boxplot:
####for the Age attributes:
```{r}
boxplot(dataset$age)
```
####for the resting blood pressure (trestbps) attributes:
```{r}
boxplot(dataset$trestbps)
```
####for the serum cholestoral (chol) attributes:
```{r}
boxplot(dataset$chol)
```
####for the maximum heart rate achieveddl (thalach) attributes:
```{r}
boxplot(dataset$thalach)
```
####for the ST depression (oldpeak) attributes:
```{r}
boxplot(dataset$oldpeak)
```
####for the number of major vessels (ca) attributes:
```{r}
boxplot(dataset$ca)
```
###Graphical representation:

```{r}
p2 <- ggplot(dataset, aes(y = sex, fill = target)) +
    geom_bar(position = "fill", width = 0.6, show.legend = TRUE) + 
    scale_y_discrete(labels  = c("0 (Female)", "1 (Male)"))+
    labs(y = "Sex", x="")+
    theme(text = element_text(size = 20))   
options(repr.plot.width = 20, repr.plot.height = 7)
(p1+p2)
```


