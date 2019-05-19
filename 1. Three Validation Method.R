### 3 Ways To Cross Validation

### 0. Data Import & Basic Statistics
#install.packages("doMC", repos="http://R-Forge.R-Project.org")
library(caret)


# Auto data 분포 확인
df <- data.frame(data=Auto)
plot(df$data.horsepower, df$data.mpg)


# 기초통계량 확인
dim(df)
str(df)
head(df, 5)
summary(df)
apply(df, 2, function(x) length(unique(x)))


### 1.validation set approach
for (j in 1:10){
  set.seed(10*j)
  train=sample(392,196)
  vs.error=rep(0,10)
  for(i in 1:10){
    fit<-knnreg(mpg~horsepower,data=Auto,k=i)
    vs.error[i]=mean((Auto$mpg[-train]-predict(fit,Auto[-train,]))^2)
  }
  if (j<2) {
    plot(vs.error, type="l", ylim=c(10,30), ylab="MSE", xlab="Number of K")
  } else {lines(vs.error, type="l", col=j)}
}
# Validation Set Apprach 최적의 K값
which.min(vs.error)



### 2.Leave-One-Out Cross-Validation
ind<-(1:n)
nmse<-matrix(0,ncol=10,nrow=n)
nd<-10

library(doMC)
registerDoMC(10)

cvp<-foreach (i = 1:n) %dopar% {
  cvmse<-numeric(nd)
  
  for (j in 1:nd){
    fit<-knnreg(mpg~horsepower,data=Auto[ind!=i,],k=j)
    pred<-predict(fit,newdata=Auto[ind==i,])
    cvmse[j]<-mean((pred-Auto$mpg[ind==i])^2)
  }
  cvmse
}
for (i in 1:n) nmse[i,]<-cvp[[i]]
loocvResult <- apply(nmse,2,mean)

# LOOCV 최적의 K값
which.min(loocvResult)


(3) k-fold CV
k<-10
n<-dim(Auto)[1]
folds<-sample(1:k,n,replace=T,prob=rep(1/k,k)) 
ind<-(1:n)%%k+1
folds<-sample(ind,n)
kmse<-matrix(0,ncol=10,nrow=k)
nd<-10

library(doMC)
registerDoMC(10)
cvp<-foreach (i = 1:k) %dopar% {
  cvmse<-numeric(nd)
  for (j in 1:nd){
    fit<-knnreg(mpg~horsepower,data=Auto[folds!=i,],k=j)
    pred<-predict(fit,newdata=Auto[folds==i,])
    cvmse[j]<-mean((pred-Auto$mpg[folds==i])^2)	
  }
  cvmse
}
for (i in 1:k) kmse[i,]<-cvp[[i]]
kfoldResult <- apply(kmse,2,mean)

# K-Fold 최적의 K값
which.min(kfoldResult)
