'''
2. Use the attached MNIST_train_small.csv training data and MNIST_test_small.csv test data to do the followings (y is a categorical response with 10 classes):
  
a. Construct a classification tree, plot the tree, and report the estimate of test classification error. Draw a confusion matrix with the test set.

b. Construct a pruned classification tree, plot the tree, and report the estimate of test classification error. Draw a confusion matrix with the test set.

c. Construct a bagged ensemble model and report the estimate of test classification error. Draw a confusion matrix with the test set. Report important predictors to predict the response.

d. Construct a random forest model and report the estimate of test classification error. Draw a confusion matrix with the test set. Report important predictors to predict the response.
'''


### MNIST 데이터 임포트
train <- read.csv("MNIST_train_small.csv", header=T)
test <- read.csv("MNIST_test_small.csv", header=T)
str(train)
str(test)


# 데이터 범주형으로 변환
train$y <- as.factor(train$y)
test$y <- as.factor(test$y)
str(train$y)


### A. Classification Tree
library(rpart)
clfTree <-rpart(y ~ ., data=train, method="class")

library(rpart.plot)
prp(clfTree, type=4, extra='auto', digits=3)

printcp(clfTree)


# a-1. Regression Tree Plot
yhat <- predict(clfTree, newdata = test, type='class')

length(test$y)
length(yhat)

head(yhat)
accTable <- table(yhat, test$y)

# a-2. Regression Tree TestMSE
accuracy <- sum(diag(accTable))/sum(accTable) * 100
clfError <- (100-accuracy)
clfError # 32.8

# a-3. confusion matrix
library(caret)
confusionMatrix(yhat, test$y)

### B. Pruned Classification Tree
printcp(clfTree) # cp값 내림차순
cp.opt <- clfTree$cptable[which.min(clfTree$cptable[,"xerror"]),"CP"] # 0.01
prunedclfTree <- prune(clfTree, cp = cp.opt) # prunning한 분류나무

# b-1. Pruned Classification Tree plot
prp(prunedclfTree, type=4, extra='auto', digits=3)


# b-2. Pruned Clf Tree error
prunedyhat <- predict(prunedclfTree, newdata = test, type='class')
prunedaccTable <- table(prunedyhat, test$y)

prunedaccuracy <- sum(diag(prunedaccTable))/sum(prunedaccTable)
prunedclfError <- (1-prunedaccuracy)
prunedclfError # 0.328

# b-4. confusion matrix
library(caret)
confusionMatrix(prunedyhat, test$y)

### C. Bagged ensemble model
library(ipred)
baggingRegModel <- bagging(y ~ ., data = train, nbagg = sqrt(nrow(train)))


# c-1. Estimate of test classification error
yhatBaggedclf <- predict(baggingRegModel, newdata = test)

baggingaccTable <- table(yhatBaggedclf, test$y)
baggingaccTable

baggingaccuracy <- sum(diag(baggingaccTable))/sum(baggingaccTable)
baggingerror <- (1-baggingaccuracy)
baggingerror # 9.7


# c-2. Draw a confusion matrix

# c-3. Important predictors (클수록 중요한 변수)
yhatBaggedclf$variable.importance


### D. Random forest model
library(randomForest)
library(gbm)

rfclf <- randomForest(y~., data = train, mtry=1, importance=T)
rfclf


# d-1. Test MSE
yhat.rf <- predict(rfclf, newdata = test)
yhat.rf

rfaccTable <- table(yhat.rf, test$y)
rfaccTable

rfaccuracy <- sum(diag(rfaccTable))/sum(rfaccTable)
rferror <- (1-rfaccuracy)
rferror # 0.74


# d-2. Draw a confusion matrix
confusionMatrix(yhat.rf, test$y)


# d-3. important predictors (클수록 중요한 변수)
importance(rfclf)
varImpPlot(rfclf, main="varImpPlot of y")







fit <- gbm(y~., data = train, n.tree=1000, shrinkage = 0.001, interaction.depth=1, bag.fraction=0.5, cv.folds = 10, distribution='multinomial')

best.iter <- gbm.perf(fit, method = 'cv') # 그래프 보면 계속 cv error 감소하는 중 => iter 더 늘려야함
best.iter # 결과값 : 

pred <- predict(fit.test, best.iter, type = 'response')

pcl <- rep('No', length(test$y))
pcl[pred[,2,1] >= 0.5] <- 'Yes'

table(pcl, test$y)
mean(pcl != test$y)









