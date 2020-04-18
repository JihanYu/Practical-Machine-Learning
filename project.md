---
title: ""
output: html_document
---

## 1. Title : Quality of the exercise can be predicted with the wearable device.

## 2. Synopsis
  The amount of exercise can be calculated by measuring the training time. However to measure quality of the exercise is challenging because it is hard to be measured and analyzed. Recently, variable wearable devices are available, and they can measure the movement of joints. Therefore the appropriateness of exercise can be qualified, and the quality can be extimated.
  By analyzing the data from wearble devices, we can predict build a model to predict quality of exercise. By using it, one can estimate its own exercise's appropriateness.

## 3. Data analysis


### Loading & preprocessing data

```r
pml.training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header=TRUE)
pml.testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header=TRUE)

library(caret);  library(ggplot2);  library(rattle)
training <- pml.training[, c(8:160)]    # Fitbit data only

# convert variable type : factor to numeric 
factor.id <- which(sapply(training, is.factor));  factor.n <- length(factor.id)
for(i in factor.id[-factor.n]){
  training[, i] <- as.character(training[, i])
  training[, i] <- as.numeric(training[, i])
}

##### Imputation #####
# apply all NA values to zero (0)
training[is.na(training)] <- 0

##### Spliting the training data to training subset and testing subset #####
inTrain <- createDataPartition(training$classe, p=0.7, list=FALSE)
training.in <- training[inTrain,];  testing.in <- training[-inTrain,]
dim(training.in);  dim(testing.in)
```

```
## [1] 13737   153
```

```
## [1] 5885  153
```

### Tree model 

```r
##### Tree model #####
modFit.rpart <- train(classe ~ ., method="rpart", data=training.in)
print(modFit.rpart)
```

```
## CART 
## 
## 13737 samples
##   152 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa     
##   0.03326213  0.4962717  0.33719637
##   0.06164175  0.4347832  0.23874912
##   0.11453565  0.3268507  0.06442907
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was cp = 0.03326213.
```

```r
fancyRpartPlot(modFit.rpart$finalModel)
```

<img src="figure/Training - tree model-1.png" title="plot of chunk Training - tree model" alt="plot of chunk Training - tree model" style="display: block; margin: auto;" />

```r
pr.train.rpart <- predict(modFit.rpart, newdata=training.in)   # Training subset
confusionMatrix(pr.train.rpart, training.in$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3558 1103 1085 1025  382
##          B   56  925   80  397  343
##          C  282  630 1231  830  664
##          D    0    0    0    0    0
##          E   10    0    0    0 1136
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4987          
##                  95% CI : (0.4903, 0.5071)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3447          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9109  0.34801  0.51377   0.0000  0.44990
## Specificity            0.6343  0.92093  0.78785   1.0000  0.99911
## Pos Pred Value         0.4974  0.51360  0.33847      NaN  0.99127
## Neg Pred Value         0.9471  0.85481  0.88465   0.8361  0.88968
## Prevalence             0.2843  0.19349  0.17442   0.1639  0.18381
## Detection Rate         0.2590  0.06734  0.08961   0.0000  0.08270
## Detection Prevalence   0.5207  0.13111  0.26476   0.0000  0.08342
## Balanced Accuracy      0.7726  0.63447  0.65081   0.5000  0.72450
```

```r
pr.test.rpart <- predict(modFit.rpart, newdata=testing.in)     # Testing subset
confusionMatrix(pr.test.rpart, testing.in$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1522  478  502  424  142
##          B   25  361   28  171  143
##          C  123  300  496  369  302
##          D    0    0    0    0    0
##          E    4    0    0    0  495
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4884          
##                  95% CI : (0.4755, 0.5012)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3313          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9092  0.31694  0.48343   0.0000  0.45749
## Specificity            0.6329  0.92267  0.77485   1.0000  0.99917
## Pos Pred Value         0.4961  0.49588  0.31195      NaN  0.99198
## Neg Pred Value         0.9460  0.84914  0.87660   0.8362  0.89101
## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
## Detection Rate         0.2586  0.06134  0.08428   0.0000  0.08411
## Detection Prevalence   0.5213  0.12370  0.27018   0.0000  0.08479
## Balanced Accuracy      0.7710  0.61981  0.62914   0.5000  0.72833
```
- The accuracy of the model is 49.7% in the training set and 49.1% in the testing set. The accuracy of the training set is similar to one of the testing set.
- The models with method, "rf", "lda" or "nb" are also tested, but they are stopped by error.

### Prediction the classe of the pml.testing data

```r
##### Apply the tree model to pml.testing data #####
testing <- pml.testing[, c(8:160)]
factor.id <- which(sapply(testing, is.factor));  factor.n <- length(factor.id)

for(i in factor.id[-factor.n]){
  testing[, i] <- as.character(testing[, i])
  testing[, i] <- as.numeric(testing[, i])
}

testing[is.na(testing)] <- 0
predict(modFit.rpart, newdata=testing)
```

```
##  [1] C A C A A C C A A A C C C A C A A A A C
## Levels: A B C D E
```
- By applying the tree model to the testing data, the prediction of the 20 different test cases can be accomplished.
- All of the test cases are belong to "A" or "C".

### Unsupervised model - kmeans

```r
kMeans1 <- kmeans(subset(training.in, select=-c(classe)), centers=5)

res.kmeans <- kMeans1$cluster
res.kmeans[res.kmeans==1] <- "A";  res.kmeans[res.kmeans==2] <- "B";  
res.kmeans[res.kmeans==3] <- "C";  res.kmeans[res.kmeans==4] <- "D";  
res.kmeans[res.kmeans==5] <- "E"
res.kmeans <- as.factor(res.kmeans)
training.in$clusters <- res.kmeans

confusionMatrix(training.in$clusters, training.in$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2164  627 1098  343  609
##          B  564  817  306  542  770
##          C  778  998  825 1152  985
##          D   17   18   10   22   16
##          E  383  198  157  193  145
## 
## Overall Statistics
##                                           
##                Accuracy : 0.2892          
##                  95% CI : (0.2816, 0.2969)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : 0.1044          
##                                           
##                   Kappa : 0.0911          
##                                           
##  Mcnemar's Test P-Value : <2e-16          
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5540  0.30737  0.34432 0.009769  0.05743
## Specificity            0.7277  0.80305  0.65497 0.994689  0.91696
## Pos Pred Value         0.4470  0.27242  0.17412 0.265060  0.13476
## Neg Pred Value         0.8042  0.82855  0.82543 0.836678  0.81202
## Prevalence             0.2843  0.19349  0.17442 0.163937  0.18381
## Detection Rate         0.1575  0.05947  0.06006 0.001602  0.01056
## Detection Prevalence   0.3524  0.21832  0.34491 0.006042  0.07833
## Balanced Accuracy      0.6409  0.55521  0.49965 0.502229  0.48719
```
- Although the results of the training data are known("classe"), unsupevised model is applied to compare the accuracy of the models.
- The accuracy of the kmeans model is 19.4%, which is far less than the tree model.


```r
modFit.kmeans <- train(clusters ~ ., data=subset(training.in, select=-c(classe)), method="rpart")
pr.train.kmeans <- predict(modFit.kmeans, newdata=training.in)
confusionMatrix(pr.train.kmeans, training.in$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2773 1193 1296  589  842
##          B  320  411  204  501  688
##          C  813 1054  896 1162  995
##          D    0    0    0    0    0
##          E    0    0    0    0    0
## 
## Overall Statistics
##                                           
##                Accuracy : 0.297           
##                  95% CI : (0.2894, 0.3047)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : 0.0005404       
##                                           
##                   Kappa : 0.0859          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7099  0.15463  0.37396   0.0000   0.0000
## Specificity            0.6013  0.84538  0.64518   1.0000   1.0000
## Pos Pred Value         0.4143  0.19350  0.18211      NaN      NaN
## Neg Pred Value         0.8392  0.80651  0.82987   0.8361   0.8162
## Prevalence             0.2843  0.19349  0.17442   0.1639   0.1838
## Detection Rate         0.2019  0.02992  0.06523   0.0000   0.0000
## Detection Prevalence   0.4872  0.15462  0.35816   0.0000   0.0000
## Balanced Accuracy      0.6556  0.50001  0.50957   0.5000   0.5000
```

```r
pr.test.kmeans <- predict(modFit.kmeans, newdata=testing.in)
confusionMatrix(pr.test.kmeans, testing.in$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1201  525  544  215  372
##          B  120  179   76  215  285
##          C  353  435  406  534  425
##          D    0    0    0    0    0
##          E    0    0    0    0    0
## 
## Overall Statistics
##                                           
##                Accuracy : 0.3035          
##                  95% CI : (0.2918, 0.3154)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : 0.0006802       
##                                           
##                   Kappa : 0.0947          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7174  0.15716  0.39571   0.0000   0.0000
## Specificity            0.6067  0.85335  0.64046   1.0000   1.0000
## Pos Pred Value         0.4204  0.20457  0.18857      NaN      NaN
## Neg Pred Value         0.8438  0.80838  0.83387   0.8362   0.8161
## Prevalence             0.2845  0.19354  0.17434   0.1638   0.1839
## Detection Rate         0.2041  0.03042  0.06899   0.0000   0.0000
## Detection Prevalence   0.4855  0.14868  0.36585   0.0000   0.0000
## Balanced Accuracy      0.6621  0.50525  0.51809   0.5000   0.5000
```

```r
predict(modFit.kmeans, newdata=testing)
```

```
##  [1] A A A A A C C B A A A A B A C A A A A A
## Levels: A B C D E
```
- By using the result of unsupervised kmeans model rather than known variable(classe), the tree models are made to predict the classe.
- The accuracy of the model is 16.1% in the training set and 15.7% in the testing set. The accuracy of the training set is similar to one of the testing set.
- The prediction of the test cases are belong to "C", "D" or "E". The results are somewhat different to results of supervised tree model.

## 4. Summary
- By using the data of wearable device, quality of the exercise can be analyzed and it is possible to predict one's appropriateness of exercise.
- A tree models were built to predict the multi-level categorical dependent variables.
- The accuracy of the model was nearly 50% in the training file.
- The prediction of the test cases was 40% accurate in the testing file.
- The supervised tree model was more accurate than unsupervised kmeans model.
