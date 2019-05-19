'''
1. Use Auto data in ISLR package (eliminate the last variable "name"). 
Split the data into approximately 50% training and 50% test sets.

a. Construct a regression tree to predict "mpg". 
Plot the tree and report the estimate of test MSE.

b. Construct a pruned regression tree to predict "mpg". 
Plot the tree and report the estimate of test MSE.

c. Construct a bagged ensemble model to predict "mpg". 
Plot the predicted values versus true values in the test set and report the estimate of test MSE. 
Report important predictors to predict the response.

d. Construct  a random forest model to predict "mpg". 
Plot the predicted values versus true values in the test set and report the estimate of test MSE. 
Report important predictors to predict the response.
'''


### Auto 데이터 임포트
library(ISLR)
data(Auto)


# 변수에 담고 변수 갯수 확인 후 'name'칼럼 삭제
df <- Auto
dim(df) # ncol 은 칼럼갯수만
df <- df[,c(-9)]
df


# 데이터 학습용 / 테스트용 반반 나누기
set.seed(20131439)
nrow <- dim(df)[1]
train <-sample(1:nrow,nrow/2)


### a. Regression Tree Model 만들기
library(caret)
library(rpart)
my.control<-rpart.control(xval=10, cp=0, minsplit = nrow(df)*0.05)
regTree<-rpart(mpg~.,data=df, subset=train, method = 'anova', control=my.control)
regTree


# a-1. Regression Tree Plot
library(rattle)
fancyRpartPlot(regTree)

library(rpart.plot)
prp(regTree, type=4, extra='auto', digits=3)

# a-2. Regression Tree TestMSE & Plot
yhat <- predict(regTree, newdata = df[-train,], type='vector')
length(df[-train,]$mpg)
length(yhat)

mean((df[-train,]$mpg - yhat)^2) # 10.10935
plot(df[-train,]$mpg, yhat, xlab='Observed values',ylab='Fitted values',xlim=c(0,50),ylim=c(0,50))
abline(0,1)


### b. Pruned Regression Tree       http://www.dodomira.com/2016/05/29/564/
printcp(regTree)
cp.opt <- regTree$cptable[which.min(regTree$cptable[,"xerror"]),"CP"] # 0.01
prunedregTree <- prune(regTree, cp = cp.opt) # prunning한 회귀나무

# b-1. Pruned Regression Tree
prp(prunedregTree, type=4, extra='auto', digits=3)


# b-2. Pruned Tree TestMSE & Plot
prunedyhat <- predict(prunedregTree, newdata = df[-train,], type='vector')

length(df[-train,]$mpg)
length(prunedyhat)

mean((df[-train,]$mpg - prunedyhat)^2) # 10.10935
#unpruned모델과 값이 똑같은 이유는 모델이 오버피팅되지 않았기 때문?

plot(df[-train,]$mpg, prunedyhat, xlab='Observed values',ylab='Fitted values',xlim=c(0,50),ylim=c(0,50))
abline(0,1)

### c. Bagged ensemble model to predict "mpg".   https://rpago.tistory.com/56
#Plot the predicted values versus true values in the test set and report the estimate of test MSE. 
#Report important predictors to predict the response.

library(ipred)
baggingRegModel <- bagging(mpg ~ ., data = df, subset = train, nbagg = sqrt(nrow(df)))


# c-1. Plot the predicted values versus true values
yhatBaggedReg <- predict(baggingRegModel, df[-train,])

plot(df[-train,]$mpg, yhatBaggReg)
abline(0,1)


# c-2. Test MSE
mean((df[-train,]$mpg - yhatBaggedReg)^2) # 9.552602


table(yhatBaggedReg, df[-train,]$mpg)
prop.table(table(yhatBaggedReg == df[-train,]$mpg))


# c-3. important predictors (클수록 중요한 변수)
baggingRegModel$variable.importance


#### d.Random Forest Regression 
#Plot the predicted values versus true values in the test set and report the estimate of test MSE. 
#Report important predictors to predict the response.

# d-1. Plot the predicted values versus true values
library(randomForest)
library(gbm)

rfReg <- randomForest(mpg~., data = df, subset = train, mtry=3, importance=T)
rfReg

yhat.bag<-predict(rfReg, newdata = df[-train,])
yhat.bag


# d-2. Test MSE
mean((df[-train,]$mpg - yhat.bag)^2) # 7.597737


# d-3. important predictors (클수록 중요한 변수)
importance(rfReg)
varImpPlot(rfReg, main="varImpPlot of mpg")





