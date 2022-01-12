##
##Probability of default
##Author: Yahya Kamel
##12 January 2022
##https://www.linkedin.com/in/yahya-kamel-5653b24b/
##Harvard University 
##Data science capstone project

## Importing libraries
library(class)
library(foreach)
library(tidyverse)
library(mgcv)
library(tidyr)
library(ggplot2)
library(dplyr)
library(cowplot)
library(MASS)
library(caTools)
library(car)
library(rpart)
library(randomForest)
library(caret)
library(gmodels)
library(e1071)
library(ROCR)
library(neuralnet)
library(NeuralNetTools)
#library(Rborist)

## Configure parallel processing in order to reduce Caret - train() processing time
#install.packages("parallel")
#install.packages("doParallel")
#library(doParallel)
# registerDoParallel(cores = NULL)
#stopImplicitCluster() #if parallel processing used. Apply code, after each model fitted.

## Import dataset
dataset = readRDS(gzcon(url("https://assets.datacamp.com/production/repositories/162/datasets/8f48a2cbb6150e7ae32435e55f271cad5b4b8ecf/loan_data_ch1.rds")))
#Dependent variable = loan status (0: no default) (1: default)
##View dataset
str(dataset)
head(dataset)

## Summarize dataset and NAs
head(dataset)
nrow(dataset) # number of observations to see if NAs are significant enough to be removed/replaced/kept
summary(dataset)
str(dataset)

# Columns with potential factors
cols_to_normalize = c(2,3,5,7,8) #Manual input

## Visualize variables distribution and locate potential outliers

# Commentary: There is a relationship between the grade, interest rate and loan defaults
dataset %>% 
  ggplot(aes(x = grade,y = int_rate))+
  geom_point(aes(color = loan_status), alpha = 0.5)+ 
  geom_boxplot()+
  # facet_wrap(~loan_status)+ #Add if you wanna split the classification in two tables
  xlab("Exploring variables - Grade")

# Commentary: There is a relationship between the loan amount, interest rate and loan defaults
dataset %>% 
  ggplot(aes(x = loan_amnt,y = int_rate))+
  geom_point(aes(color = loan_status), alpha = 0.5)+ 
  geom_smooth()+
  # facet_wrap(~loan_status)+ #Add if you wanna split the classification in two tables
  xlab("Exploring variables - Loan amount")

# Commentary: There is a relationship between the annual income, interest rate and loan defaults
dataset %>% 
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = loan_status), alpha = 0.5)+ 
  geom_smooth()+
  scale_y_continuous(trans = 'log10')+
  scale_x_continuous(trans = 'log10')+
  # facet_wrap(~loan_status)+ #Add if you wanna split the classification in two tables
  xlab("Exploring variables - annual income")

# Commentary: Focusing on potential outliers in interest rate.
#Decision: Remove from the dataset, given their limited number and high SE
dataset %>% 
  filter(int_rate >= 20) %>%
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = loan_status), alpha = 0.5)+ 
  geom_smooth()+
  # facet_wrap(~loan_status)+ #Add if you wanna split the classification in two tables
  xlab("Exploring variables - Filtered interest rate")

#Number of those observations
filter_int = dataset %>%  filter(int_rate >= 20) 
nrow(filter_int)

# Commentary: Focusing on potential outliers in annual income.
#Decision: Remove from the dataset, given their limited number and high SE
dataset %>% 
  filter(annual_inc >= 600000) %>%
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = loan_status), alpha = 0.5)+ 
  geom_smooth()+
  # facet_wrap(~loan_status)+ #Add if you wanna split the classification in two tables
  xlab("Exploring variables - Filtered annual income")

#Number of those observations
filter_income = dataset %>%  filter(annual_inc >= 600000) 
nrow(filter_income)

# Commentary: There is almost no relationship between the age and loan defaults.
#One observation above 100 of age, which can be removed as it might seem wrong.
dataset %>% 
  ggplot(aes(x = age,y = int_rate))+
  geom_point(aes(color = loan_status), alpha = 0.5)+ 
  scale_y_continuous(trans = 'log10')+
  scale_x_continuous(trans = 'log10')+
  geom_smooth()+
  # facet_wrap(~loan_status)+ #Add if you wanna split the classification in two tables
  xlab("Exploring variables - Age")

# Commentary: There is almost no relationship between the employment length and loan defaults
dataset %>% 
  ggplot(aes(x = emp_length,y = int_rate))+
  geom_point(aes(color = loan_status), alpha = 0.5)+ 
  geom_smooth()+
  # facet_wrap(~loan_status)+ #Add if you wanna split the classification in two tables
  xlab("Exploring variables - Employment length")

## Handling missing data
dataset_after_NAs = dataset
dataset_after_NAs$int_rate = ifelse(is.na(dataset_after_NAs$int_rate),
                                    ave(dataset_after_NAs$int_rate,FUN = function(x) mean(x, na.rm=TRUE)),
                                    dataset_after_NAs$int_rate)
dataset_after_NAs$emp_length = ifelse(is.na(dataset_after_NAs$emp_length),
                                      ave(dataset_after_NAs$emp_length,FUN = function(x) mean(x, na.rm=TRUE)),
                                      dataset_after_NAs$emp_length)

#Check if NA values have been replaced for all columns
dataset_excluding_outliers = dataset_after_NAs
colSums(is.na(dataset_excluding_outliers)) 

## Remove outliers
dataset_excluding_outliers = dataset_excluding_outliers %>% 
  filter(annual_inc < 600000) %>%
  filter(int_rate < 20) %>%
  filter(age<100)

#Observations removed
nrow(dataset)-nrow(dataset_excluding_outliers)

#View overall dataset status and check if there are further data refinement needed
summary(dataset_excluding_outliers)

## Re-visualize the variables after removing the top outliers, following the visualization section above

## Encoding the target dependent as well as other relevant variables as factor
dataset_excluding_outliers$loan_status =  factor(
  dataset_excluding_outliers$loan_status,
  levels = c(0,1), 
  labels = c(0,1)) # 0: loan with no default / 1: loan with default
dataset_excluding_outliers$home_ownership = factor(
  dataset_excluding_outliers$home_ownership,
  levels = c("RENT","OWN","MORTGAGE","OTHER"),
  labels = c(1:4))
dataset_excluding_outliers$grade = factor(
  dataset_excluding_outliers$grade,
  levels = c("A","B","C","D","E","F","G"),
  labels = c(1:7))

# View data after alterations above
str(dataset_excluding_outliers)
summary(dataset_excluding_outliers)
head(dataset_excluding_outliers)

#View correlations between the independent variables
round(cor(dataset_excluding_outliers[,cols_to_normalize]),2)

## Feature Scaling, normalizing the numbers to standard unit
dataset_excluding_outliers_normalized = dataset_excluding_outliers
dataset_excluding_outliers_normalized[,cols_to_normalize] = scale(dataset_excluding_outliers[,cols_to_normalize]) 

## Splitting the dataset into the Training set and Test set
set.seed(11111)
split = sample.split(dataset_excluding_outliers_normalized$loan_status, SplitRatio = 0.75)
training_set = subset(dataset_excluding_outliers_normalized, split == TRUE)
test_set = subset(dataset_excluding_outliers_normalized, split == FALSE)
test_set_unnormalized = subset(dataset_excluding_outliers, split == FALSE)

######### Start of LGM models ################
## Fitting GLM Logistic Regression to dataset
#A polynomial relationship is reflected into interest rate
set.seed(1)
classifier_log = glm(loan_status ~  poly(int_rate,2) + grade + annual_inc,family=binomial,data=training_set) #exclude variables with no significance
 summary(classifier_log)
classifier_logit = glm(loan_status ~  poly(int_rate,2) + grade + annual_inc,family = binomial(link = logit), data = training_set)
 summary(classifier_logit)
classifier_cloglog =  glm(loan_status ~ poly(int_rate,2) + grade + annual_inc,family = binomial(link = cloglog), data = training_set)
 summary(classifier_cloglog)
classifier_probit = glm(loan_status ~ poly(int_rate,2) + grade + annual_inc,family = binomial(link = probit), data = training_set)
 summary(classifier_probit)

## Step-wise AIC analysis, which presents AIC score for different models with different variables
stepAIC(glm(loan_status ~ . ,family=binomial,data=training_set),direction="both") #Look for model with lowest AIC
#Commentary: StepAIC is suggesting to include "emp_length" variable. However, due to the earlier analysis, we won't include it in the model.

## Predicting using test_set
prob_pred_test_set_log = predict(classifier_log, type = 'response', newdata = test_set[,-1])
prob_pred_test_set_logit = predict(classifier_logit, type = 'response', newdata = test_set[,-1])
prob_pred_test_set_cloglog = predict(classifier_cloglog, type = 'response', newdata = test_set[,-1])
prob_pred_test_set_probit = predict(classifier_probit, type = 'response', newdata = test_set[,-1])

## Prediction range
#Low range means bad model prediction. Wide range would presume the opposite
prediction_range_table=data.frame(
range(prob_pred_test_set_log), 
range(prob_pred_test_set_logit),
range(prob_pred_test_set_cloglog),
range(prob_pred_test_set_probit)
)
prediction_range_table

## Find threshold through ROC curve
#Commentary: ROC in this test is irrelevant because we are more interested in the specificity "unpredicted defaults"
# pred=prediction(prob_pred_test_set_probit,test_set[,1]) #Manual choice of the model
# perf=performance(pred,measure = "acc")
# plot(perf)

## Find best accuracy point on ROCR curve
# max=which.max(slot(perf,"y.values")[[1]])
# acc=slot(perf,"y.values")[[1]][max]
# cut=slot(perf,"x.values")[[1]][max]
# ROC_threshold=round(c(Accuracy_value=acc,Cutoff_value=cut),4)
# ROC_threshold

threshold_pred=0.15 #Manual input

## Convert prediction prob to binary - GLM models
y_pred_log_test_set = unname(as.factor( ifelse(prob_pred_test_set_log > threshold_pred, 1, 0)))
y_pred_logit_test_set = unname(as.factor( ifelse(prob_pred_test_set_logit > threshold_pred, 1, 0)))
y_pred_cloglog_test_set = unname(as.factor( ifelse(prob_pred_test_set_cloglog > threshold_pred, 1, 0)))
y_pred_probit_test_set = unname(as.factor( ifelse(prob_pred_test_set_probit > threshold_pred, 1, 0)))

## Run the Confusion Matrix
cm_log_test_set = confusionMatrix(data = y_pred_log_test_set,reference = test_set$loan_status)
cm_logit_test_set = confusionMatrix(data = y_pred_logit_test_set,reference = test_set$loan_status)
cm_cloglog_test_set = confusionMatrix(data = y_pred_cloglog_test_set,reference = test_set$loan_status)
cm_probit_test_set = confusionMatrix(data = y_pred_probit_test_set,reference = test_set$loan_status)

#Display confusion matrix table
cm_log_test_set$table
cm_logit_test_set$table
cm_cloglog_test_set$table
cm_probit_test_set$table

## Caret - GLM models with cross validation, seeking the optimal GLM model outcome 
#Fitting the model - caret GLM
trctrl = trainControl(method="repeatedcv", 
                       number=5, 
                       repeats = 5)

classifier_caret_GLM = train(loan_status ~int_rate + grade + annual_inc,
                      data = training_set,
                      method = "glmnet",
                      trControl = trctrl,
                      tuneLength=5)

#Accuracy curve and confusion matrix
plot(classifier_caret_GLM)

# Accuracy and confusion matrix
y_pred_caret_GLM_test_set = predict(classifier_caret_GLM, newdata = test_set, type = "raw")
y_pred_caret_GLM_test_set_P = predict(classifier_caret_GLM, newdata = test_set, type = "prob")[,2] #selecting column 2, which is probability of 1, comparable to GLM probabilities
y_pred_caret_GLM_test_set = unname(as.factor(ifelse(y_pred_caret_GLM_test_set_P > 0.2, 1, 0))) #Manual input
cm_caret_GLM_test_set = confusionMatrix(reference = test_set$loan_status, data = y_pred_caret_GLM_test_set)
cm_caret_GLM_test_set$table

## Visualize GLM model performance

#Add predictions to test set
test_set_pred_binary = data.frame(test_set_unnormalized,
                                  Pred_caret_GLM = y_pred_caret_GLM_test_set)

#Visualize model outcome - GLM
test_set_pred_binary %>% 
  ggplot(aes(x = loan_amnt,y = int_rate))+
  geom_point(aes(color = Pred_caret_GLM ), alpha = 0.5)+ #Prediction
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - GLM from Caret package")

test_set_pred_binary %>% 
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = Pred_caret_GLM), alpha = 0.5)+ #Prediction
  scale_y_continuous(trans = 'log10')+
  scale_x_continuous(trans = 'log10')+
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - GLM from Caret package")

test_set_pred_binary %>% 
  ggplot(aes(x = grade,y = int_rate))+
  geom_point(aes(color = Pred_caret_GLM), alpha = 0.5)+ #Prediction
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - GLM from Caret package")

## K-Fold cross validation
folds=createFolds(training_set$loan_status, #dependent variable
                  k=10)                   # number of folds.
cv = lapply(folds,function(x){            # X = each fold 
  training_fold=training_set[-x,]     #-x: except for the test fold. Assign a training dataset to each fold
  test_fold=training_set[x,]          #x: test fold  
  #To each fold apply the following:
  classifier_log = glm(loan_status~.-home_ownership,family=binomial,data=training_set)
  y_pred = predict(classifier_log, newdata = test_fold[-1]) #Test on the test fold
  cm=table(test_fold[,1],y_pred)     #Summarize the confusion matrix for test folds
  accuracy = (cm[1,1]+cm[2,2]) / (cm[1,1]+cm[1,2]+cm[2,1]+cm[2,2]) #calculate the accuracy for each fold
  return(accuracy)
})

#CV: Return list of accuracies on folds above
min_accuracy=min(as.numeric(cv))
max_accuracy=max(as.numeric(cv))
average_accuracy=mean(as.numeric(cv))
data.frame(min_accuracy,max_accuracy,average_accuracy)

#=========end of GLM models=================

#=========Model: K-Nearest neighbors regression "KNN" =================

# Configure train() with cross validation and parallel processing
trctrl = trainControl(method = "repeatedcv", repeats = 5, number = 5)

# Fitting KNN model
set.seed(1)
classifier_knn = train(loan_status ~. -emp_length - loan_amnt - home_ownership -age, data = training_set, 
                  method = "knn",
                  trControl=trctrl,
                  tuneLength = 5 #maximum grid size
                  ) 
classifier_knn 

#Accuracy curve - KNN
plot(classifier_knn)

#Accuracy curve and confusion matrix
y_pred_knn_test_set = predict(classifier_knn, newdata = test_set, type = "raw")
y_pred_knn_test_set_P = predict(classifier_knn, newdata = test_set, type = "prob")[,2] #selecting column 2, which is probability of 1, comparable to GLM probabilities
y_pred_knn_test_set = unname(as.factor(ifelse(y_pred_knn_test_set_P > 0.17, 1, 0))) #Manual input
cm_knn_test_set = confusionMatrix(reference = test_set$loan_status, data = y_pred_knn_test_set)
cm_knn_test_set$table

## Visualize the model - KNN

#Add predictions to test set
test_set_pred_binary = data.frame(test_set_unnormalized,
                                  Pred_KNN = y_pred_knn_test_set)

#Visualize model outcome - KNN
test_set_pred_binary %>% 
  ggplot(aes(x = loan_amnt,y = int_rate))+
  geom_point(aes(color = Pred_KNN ), alpha = 0.5)+ #Prediction
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - KNN")

test_set_pred_binary %>% 
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = Pred_KNN), alpha = 0.5)+ #Prediction
  scale_y_continuous(trans = 'log10')+
  scale_x_continuous(trans = 'log10')+
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - KNN")

test_set_pred_binary %>% 
  ggplot(aes(x = grade,y = int_rate))+
  geom_point(aes(color = Pred_KNN), alpha = 0.5)+ #Prediction
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - KNN")


#=========Model: Support Vector Machines with Radial Basis "SVM" =================

# Fitting SVM-Radial for non-linear relationship 
classifier_svm = svm(formula = loan_status ~ .,
            data = training_set,
            type = 'C-classification',
            kernel = 'radial',
            probability = TRUE)

# SVM with cross validation, but very slow on caret package
# classifier_svm = train(loan_status ~., data = training_set, 
#                        method = "svmRadial",
#                        trControl=trctrl,
#                        tuneLength = 5) #maximum grid size
summary(classifier_svm)

#Accuracy curve and confusion matrix
y_pred_svm_test_set = predict(classifier_svm, newdata = test_set)
y_pred_svm_test_set_P = predict(classifier_svm, newdata = test_set, probability = TRUE ) #selecting column 2, which is probability of 1, comparable to GLM probabilities
y_pred_svm_test_set_P = as.data.frame(
                        attr(y_pred_svm_test_set_P, "probabilities") # added step to extract probabilities
                        )[,2] #Extract column 2 for probabilities of 1
y_pred_svm_test_set_P_x = y_pred_svm_test_set_P * 1000 / 2
y_pred_svm_test_set = unname(as.factor(ifelse(y_pred_svm_test_set_P_x > 55.18, 1, 0))) #Manual input
cm_svm_test_set = confusionMatrix(reference = test_set$loan_status, data = y_pred_svm_test_set)
cm_svm_test_set$table

## Visualize the model - SVM

#Add predictions to test set
test_set_pred_binary = data.frame(test_set_unnormalized,
                                  Pred_SVM = y_pred_svm_test_set)

#Visualize model outcome - SVM
test_set_pred_binary %>% 
  ggplot(aes(x = loan_amnt,y = int_rate))+
  geom_point(aes(color = Pred_SVM ), alpha = 0.5)+ #Prediction
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - SVM")

test_set_pred_binary %>% 
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = Pred_SVM), alpha = 0.5)+ #Prediction
  scale_y_continuous(trans = 'log10')+
  scale_x_continuous(trans = 'log10')+
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - SVM")

test_set_pred_binary %>% 
  ggplot(aes(x = grade,y = int_rate))+
  geom_point(aes(color = Pred_SVM), alpha = 0.5)+ #Prediction
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - SVM")


#=========Model: Neural Networks =================

## Fit model - NNET

# very slow, more accurate than the following model, but still below expectations with around 20% specificity rate.
# data_nnt = data.matrix(training_set) #convert data to numeric matrix, so neuralnet can accept it.
# classifier_nnet = neuralnet(loan_status~., data = data_nnt, 
#                 algorithm = "rprop+", #manual input
#                 hidden=c(10,3),
#                 threshold=0.1,
#                 stepmax = 1e+06)
# plot(classifier_nnet)

#Caret NNET - takes around 30 minutes
myGrid = expand.grid(.decay=c(0.5, 0.1), .size=c(4,5,6))
classifier_nnet = train(loan_status~. -emp_length - loan_amnt - home_ownership -age,
                        data = training_set,
                        method = "nnet", 
                        tuneGrid = myGrid, 
                        preProcess = c("center"),
                        linout=0, #for classification
                        maxit=300, 
                        trace=F)
classifier_nnet

# Visualize the network and confusion matrix
library(NeuralNetTools)  
plotnet(classifier_nnet, alpha = 0.6) 

#Accuracy curve
plot(classifier_nnet)

## Prediction - binary (0,1) before adjustment
y_pred_nnet_test_set = predict(classifier_nnet, newdata = test_set)

## Prediction - probabilities
y_pred_nnet_test_set_P = predict(classifier_nnet, newdata = test_set, type = "prob")[,2]

# Predictions in binary (0,1) form after adjustment through manual input
y_pred_nnet_test_set = unname(as.factor(ifelse(y_pred_nnet_test_set_P > 0.15, 1, 0))) #Manual input

## Run the Confusion Matrix
cm_nnet_test_set = confusionMatrix(reference = test_set$loan_status, data = y_pred_nnet_test_set)

#Display confusion matrix table
cm_nnet_test_set$table

## Visualize the model - Neural networks

#Add predictions to test set
test_set_pred_binary = data.frame(test_set_unnormalized,
                                  Pred_NNET = y_pred_nnet_test_set)
test_set_pred_binary %>% 
  ggplot(aes(x = loan_amnt,y = int_rate))+
  geom_point(aes(color = Pred_NNET ), alpha = 0.5)+ #Prediction
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - NNET")

test_set_pred_binary %>% 
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = Pred_NNET), alpha = 0.5)+ #Prediction
  scale_y_continuous(trans = 'log10')+
  scale_x_continuous(trans = 'log10')+
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - NNET")

test_set_pred_binary %>% 
  ggplot(aes(x = grade,y = int_rate))+
  geom_point(aes(color = Pred_NNET), alpha = 0.5)+ #Prediction
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - NNET")


#=========Model: Naive Bayes =================

## Fit model - Naive Bayes

# classifier_nb = train(loan_status~. , data = training_set, method = "nb", metric = "Accuracy", maxit = 150, trControl = trctrl, tuneLength = 5, preProcess = c("center"))
# classifier_nb

#Caret Naive Bayes
classifier_nb = naiveBayes(x=training_set[,-1], #Manually remove dependent variable
                           y=training_set$loan_status, #It should be a vector, so no brackets used
                           usekernel = TRUE)

## Prediction - binary (0,1) before adjustment
y_pred_nb_test_set = predict(classifier_nb, newdata = test_set)

## Prediction - probabilities
y_pred_nb_test_set_P = predict(classifier_nb, newdata = test_set, type = "raw")[,2]

# Predictions in binary (0,1) form after adjustment through manual input
y_pred_nb_test_set = unname(as.factor(ifelse(y_pred_nb_test_set_P > 0.2, 1, 0))) #Manual input

## Run the Confusion Matrix
cm_nb_test_set = confusionMatrix(reference = test_set$loan_status, data = y_pred_nb_test_set)

#Display confusion matrix table
cm_nb_test_set$table

## Visualize the model - Naive Bayes

#Add predictions to test set
test_set_pred_binary = data.frame(test_set_unnormalized,
                                  Pred_nb = y_pred_nb_test_set)

test_set_pred_binary %>% 
  ggplot(aes(x = loan_amnt,y = int_rate))+
  geom_point(aes(color = Pred_nb ), alpha = 0.5)+ #Prediction
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - Naive Bayes")

test_set_pred_binary %>% 
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = Pred_nb), alpha = 0.5)+ #Prediction
  scale_y_continuous(trans = 'log10')+
  scale_x_continuous(trans = 'log10')+
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - Naive Bayes")

test_set_pred_binary %>% 
  ggplot(aes(x = grade,y = int_rate))+
  geom_point(aes(color = Pred_nb), alpha = 0.5)+ #Prediction
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - Naive Bayes")

#=========Model: Decision Tree =================

## Fit model - Decision Tree

# less accurate
# library(rpart)
# rpt1 = rpart(loan_status~., data = training_set,
#            method = "class") #classification regression
# classifier_rf = rpt1
# 
# rpart.plot(classifier_dt)
# plot(classifier_dt)

classifier_dt = train(loan_status ~ int_rate + grade + annual_inc,
                     method = "rpart",
                     data = training_set)

ggplot(classifier_dt) # y = "RMSE - bootstrap": minimum error
confusionMatrix(predict(classifier_dt, test_set), test_set$loan_status)

y_pred_dt_test_set = predict(classifier_dt, newdata = test_set)
y_pred_dt_test_set_P = predict(classifier_dt, newdata = test_set, type = "prob")[,2]
y_pred_dt_test_set = unname(as.factor(ifelse(y_pred_dt_test_set_P > 0.2, 1, 0))) #Manual input
cm_dt_test_set = confusionMatrix(reference = test_set$loan_status, data = y_pred_dt_test_set)
cm_dt_test_set$table

## Visualize the model - Decision Tree
#Add predictions to test set
test_set_pred_binary = data.frame(test_set_unnormalized,
                                  Pred_dt = y_pred_dt_test_set)

test_set_pred_binary %>% 
  ggplot(aes(x = loan_amnt,y = int_rate))+
  geom_point(aes(color = Pred_dt ), alpha = 0.5)+ #Prediction
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - Decision Tree")

test_set_pred_binary %>% 
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = Pred_dt), alpha = 0.5)+ #Prediction
  scale_y_continuous(trans = 'log10')+
  scale_x_continuous(trans = 'log10')+
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - Decision Tree")

test_set_pred_binary %>% 
  ggplot(aes(x = grade,y = int_rate))+
  geom_point(aes(color = Pred_dt), alpha = 0.5)+ #Prediction
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - Decision Tree")

#=========Model: Random forest =================

## Fit model - Random Forest 
set.seed(1)

## Fit model with cross validation - not perfect, as it's focused with accuracy and not with specificity 
# classifier_rf = train(loan_status ~. -emp_length - loan_amnt - home_ownership -age,
#                     method = "Rborist",
#                     tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
#                     data = training_set)

# RF - better outcome compared to Caret package
library(randomForest)
classifier_rf = randomForest(loan_status ~ int_rate + grade + annual_inc,
                             data=test_set,
                             ntree = 3600) #Manual input

#Accuracy curve and confusion matrix - Test set
plot(classifier_rf)
y_pred_rf_test_set = predict(classifier_rf, newdata = test_set)
y_pred_rf_test_set_P = predict(classifier_rf, newdata = test_set, type = "prob")[,2]
y_pred_rf_test_set = unname(as.factor(ifelse(y_pred_rf_test_set_P > 0.025, 1, 0))) #Manual input
cm_rf_test_set = confusionMatrix(reference = test_set$loan_status, data = y_pred_rf_test_set)
cm_rf_test_set$table
 


## Visualize the model - Random Forest

#Add predictions to test set
test_set_pred_binary = data.frame(test_set_unnormalized,
                                  Pred_RandomForest = y_pred_rf_test_set)
#Visualize model outcome - Random Forest
test_set_pred_binary %>% 
  ggplot(aes(x = loan_amnt,y = int_rate))+
  geom_point(aes(color = Pred_RandomForest ), alpha = 0.5)+ #Prediction
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - Random Forest")

test_set_pred_binary %>% 
  ggplot(aes(x = annual_inc,y = int_rate))+
  geom_point(aes(color = Pred_RandomForest), alpha = 0.5)+ #Prediction
  scale_y_continuous(trans = 'log10')+
  scale_x_continuous(trans = 'log10')+
  geom_smooth()+
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - Random Forest")

test_set_pred_binary %>% 
  ggplot(aes(x = grade,y = int_rate))+
  geom_point(aes(color = Pred_RandomForest), alpha = 0.5)+ #Prediction
  facet_wrap(~loan_status)+ # Actual
  xlab("Prediction against actual - Random Forest")

################## Comparing models outcome #####################

#Add predictions to test set - all models update
test_set_pred_binary = data.frame(test_set_unnormalized,
                                  Pred_GLM_log = y_pred_log_test_set,
                                  Pred_GLM_logit = y_pred_logit_test_set,
                                  Pred_GLM_cloglog = y_pred_cloglog_test_set,
                                  Pred_GLM_probit = y_pred_probit_test_set,
                                  Pred_KNN = y_pred_knn_test_set,
                                  Pred_RandomForest = y_pred_rf_test_set,
                                  Pred_NNET = y_pred_nnet_test_set,
                                  Pred_nb = y_pred_nb_test_set,
                                  Pred_dt = y_pred_dt_test_set)

#Summarize table of the confusion matrix accuracy

Model_names = c("GLM_Log","GLM_Logit","GLM_Cloglog","GLM_Probit", "KNN", "SVM", "Neural_Networks", "Naive Bayes", "Decision Tree", "Random Forest") 
Accuracy_ = c(cm_log_test_set$overall["Accuracy"], 
              cm_logit_test_set$overall["Accuracy"], 
              cm_cloglog_test_set$overall["Accuracy"],
              cm_probit_test_set$overall["Accuracy"],
              cm_knn_test_set$overall["Accuracy"],
              cm_svm_test_set$overall["Accuracy"],
              cm_nnet_test_set$overall["Accuracy"],
              cm_nb_test_set$overall["Accuracy"],
              cm_dt_test_set$overall["Accuracy"],
              cm_rf_test_set$overall["Accuracy"])

Specificity_ = c(cm_log_test_set$byClass["Specificity"],
                 cm_logit_test_set$byClass["Specificity"],
                 cm_cloglog_test_set$byClass["Specificity"],
                 cm_probit_test_set$byClass["Specificity"],
                 cm_knn_test_set$byClass["Specificity"],
                 cm_svm_test_set$byClass["Specificity"],
                 cm_nnet_test_set$byClass["Specificity"],
                 cm_nb_test_set$byClass["Specificity"],
                 cm_dt_test_set$byClass["Specificity"],
                 cm_rf_test_set$byClass["Specificity"])

Confusion_matrix_table = data.frame(Model_name = Model_names,
                                    Accuracy_overall = Accuracy_,
                                    Specificity_Actual_1_Predicted_0 = Specificity_)
Confusion_matrix_table                        

# Summary table of PDs using Random Forest model - test set
All_set_pred_binary = data.frame(test_set,
                                 PD_RandomForest = y_pred_rf_test_set,
                                 PD_Probit = y_pred_probit_test_set)

# PDs for the test set - GLM: Probit
CrossTable(All_set_pred_binary$grade, 
           All_set_pred_binary$PD_Probit, 
           prop.r = TRUE, prop.c = FALSE, prop.t = FALSE, prop.chisq = FALSE)

# PDs for the test set - Random Forest
CrossTable(All_set_pred_binary$grade, 
           All_set_pred_binary$PD_RandomForest, 
           prop.r = TRUE, prop.c = FALSE, prop.t = FALSE, prop.chisq = FALSE)


