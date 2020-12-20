# Data Dictionary
# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded
### Note that ip, app, device, os, and channel are encoded.
# The test data is similar, with the following differences:
# click_id: reference for making predictions
# is_attributed: not included


# Loading library
#install.packages("markdown")
# install.packages('caret')
# install.packages('data.table')
# install.packages("corrplot")
# install.packages('e1071')
# install.packages('ROSE')
library(markdown)
library(ROSE)
library(corrplot)
library(data.table)
library(caret)
library(e1071)
library(dplyr)
# Loading datasets

train <- fread("talkingdata-adtracking-fraud-detection/train.csv")
test <- fread("talkingdata-adtracking-fraud-detection/test.csv")
sample_subm <- fread("talkingdata-adtracking-fraud-detection/sample_submission.csv")

#Data Exploration
str(train)
str(test)
str(sample_subm)

View(train)
View(test)
View(sample_subm)
#Table
prop.table(table(train$is_attributed))

#Data Split
set.seed(283)
indextrain <- sample(nrow(train), size = 0.0001 * nrow(train))
indextest <- sample(nrow(test), size = 0.001 * nrow(test))


df_train <- train[indextrain,]
str(df_train)

df_test <- test[indextest,]
str(df_test)

#Removing click_time & attributed_time
df_train$click_time <- NULL
df_test$click_time <- NULL
df_train$attributed_time <- NULL

#Correlation between features
cor(df_train)
corrplot(cor(df_train), is.corr = F )

cor(df_test)
corrplot(cor(df_test), is.corr = F)

#Removing IP & click_id
df_train$ip <- NULL
df_test$ip <- NULL
df_test$click_id <- NULL


str(df_test)
str(df_train)

#Table of target feature
table(df_train$is_attributed)

#Applying undersampling.
# Using Under Both or Over
df_train1 <- ovun.sample(is_attributed ~., data = df_train, method = 'both', N = 45000)$data
View(df_train1)
table(df_train1$is_attributed)

#Function for automating variable categorization
catfun <- function(dataset, features){
  for (feature in features) {
    dataset[[feature]] <- as.factor(dataset[[feature]])
  }
  return(dataset)
}

#Function for  Normalization
catnorm <- function(dataset, features){
  for(feature in features){
    dataset[[feature]] <- scale(dataset[[feature]], center = T, scale = T)
  }
  return(dataset)
}

#Features
testnorm <- c('channel', 'os', 'device', 'app')
trainnorm <- c('app', 'device', 'os', 'channel')
traincat <- c('is_attributed')

# Categorization
df_train <- catfun(df_train1, trainf)

str(df_train)

# Normalization
df_train <- catnorm(df_train, trainnorm)
df_test <- catnorm(df_test, testnorm)

str(df_train)
str(df_test)
#Model training
modelv1 <- train(is_attributed ~., data = df_train, method = 'rf')
modelv1

predc <- predict(modelv1, df_test)
View(predc)

#Submission
View(predc)
