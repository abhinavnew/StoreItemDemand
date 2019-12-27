
library(caret)
library(tidyr)
library(plyr)
library(dplyr)
library(caTools)
library(reshape2)
library(gbm)
library(caTools)
library(randomForest)
library(ggplot2)
library(data.table)
library(xgboost)
library(Matrix)
library(lightgbm)
library(TeachingDemos)
library(Hmisc)
library(DMwR)
library(e1071)
library(glmnet)
library(pROC)
library(funModeling)
library(lubridate)


##Notations off and clear all objects
options(scipen = 999)
rm(list = ls())
gc()

starttime=Sys.time()

zerovari <- function(dat) {
  out <- lapply(dat, function(x) length(unique(x)))
  want <- which(!out > 1)
  unlist(want)
}

first_day_of_month_wday <- function(dt) {
  day(dt) <- 1
  wday(dt)
}

lift <- function(depvar, predcol, groups=10) {
  if(!require(dplyr)){
    install.packages("dplyr")
    library(dplyr)}
  if(is.factor(depvar)) depvar <- as.integer(as.character(depvar))
  if(is.factor(predcol)) predcol <- as.integer(as.character(predcol))
  helper = data.frame(cbind(depvar, predcol))
  helper[,"bucket"] = ntile(-helper[,"predcol"], groups)
  gaintable = helper %>% group_by(bucket)  %>%
    summarise_at(vars(depvar), funs(total =n(),
                                    totalresp=sum(., na.rm = TRUE))) %>%
    mutate(Cumresp = cumsum(totalresp),
           Gain=Cumresp/sum(totalresp)*100,
           Cumlift=Gain/(bucket*(100/groups)))
  return(gaintable)
}

storedata_origtrain=fread("E:\\AbhinavB\\Kaggle\\StoreItemDemandPrediction\\train.csv",
                   data.table = FALSE,stringsAsFactors = TRUE)
#storedata_origtrain=read_csv("../input/demand-forecasting-kernels-only/train.csv")

storedata_origtest=fread("E:\\AbhinavB\\Kaggle\\StoreItemDemandPrediction\\test.csv",data.table = FALSE,stringsAsFactors = TRUE)
#storedata_origtest=read_csv("../input/test.csv")



dim(storedata_origtrain)
dim(storedata_origtest)

storedata_origtrain$id=NA
storedata_origtrain$ind="train"
storedata_origtest$sales=NA
storedata_origtest$ind="test"
combined=rbind(storedata_origtrain,storedata_origtest)
dim(combined)
glimpse(combined)
                   

storedata=combined
dim(storedata)

##date type feature creation 

storedata$date2=ymd(storedata$date)
storedata$YEAR=lubridate::year(storedata$date2)
storedata$MNTH=lubridate::month(storedata$date2)
storedata$MNTH_NM=lubridate::month(storedata$date2,label=TRUE)
##Day of week month year 
storedata$DayOfWeek=lubridate::wday(storedata$date2)
storedata$DayOfWeek_Name=lubridate::wday(storedata$date2,label=TRUE)
storedata$DayOfMonth=lubridate::mday(storedata$date2)
storedata$DayOfYear=lubridate::yday(storedata$date2)
##Week of year & month
storedata$WeekOfYr=lubridate::week(storedata$date2)
storedata$WeekOfMnth=ceiling((day(storedata$date2)+first_day_of_month_wday(storedata$date2)-1)/7)
dim(storedata)

##volume of items sold related feature creation 
##resplit into test and train/validate set 



testset=storedata[storedata$ind=="test",]
dim(testset)

train=storedata[storedata$ind=="train",]
dim(train)
print("test and train deMerged")


valset=train[train$date2>=as.Date("2017-01-01"),]
dim(valset)
trainset=train[train$date2<as.Date("2017-01-01"),]
dim(trainset)

##xgboost training set prep


trainset=trainset[,-which(names(trainset) %in% c("id","ind","date","MNTH_NM","DayOfWeek_Name","date2"))]
caretset=trainset
tr_labels=trainset[,"sales"]
trainset$sales<-NULL
x=as.matrix(trainset)
head(x)
x=matrix(as.numeric(x),nrow(x),ncol(x))
dtrain=xgb.DMatrix(data=x,label=tr_labels,missing = NA)
dim(dtrain)

valset=valset[,-which(names(valset) %in% c("id","ind","date","MNTH_NM","DayOfWeek_Name","date2"))]
caretvalset=valset
vl_labels=valset[,"sales"]
valset$sales<-NULL

y=as.matrix(valset)
y=matrix(as.numeric(y),nrow(y),ncol(y))
dval=xgb.DMatrix(data=y,label=vl_labels,missing = NA)
dim(dval)

ts_labels=testset[,"sales"]
save_ids=testset[,"id"]
testset$sales<-NULL
testset=testset[,-which(names(testset) %in% c("id","ind","date","MNTH_NM","DayOfWeek_Name","date2"))]
z=as.matrix(testset)
z=matrix(as.numeric(z),nrow(z),ncol(z))
dtest=xgb.DMatrix(data=z,label=ts_labels,missing = NA)
dim(dtest)

param=list(booster="gbtree",
           objective="reg:linear",
           eval_metric="mae",
           eta=0.3
           )
# ## max_depth=6,
# gamma=0,
# colsample_bytree=0.6,
# min_chld_weight=1,
# subsample=1
xgcv=xgb.cv(params = param,data=dtrain,nrounds = 1000,nfold = 5,showsd = T,stratified = T,print_every_n = 10,early_stopping_rounds = 20,maximize = F)
bstiter=xgcv$best_iteration
bstiter

##,subsample=0.5,colsample_bytree=0.5,

watcher=list(train=dtrain)



##ctrlobj=trainControl(method="cv",number = 3,allowParallel =T )

##tgrid=expand.grid(eta=c(0.01,0.05,0.1),subsample=c(0.5,1),colsample_bytree=c(0.4,0.6),nrounds=1000,max_depth=c(4,6),min_child_weight=1,gamma=0)

set.seed(51219)
##xgb_mod1=train(sales~.,data=caretset,method="xgbTree",trControl=ctrlobj,tuneGrid=tgrid,verbose=T,metric="RMSE")


xgb_mod1=xgb.train(params = param,data=dtrain,watchlist = watcher,nrounds=bstiter,verbose = 1)


summary(xgb_mod1)

predval=predict(xgb_mod1,newdata = dval)

head(predval)

sm=smape(vl_labels,predval)
sm
print("smape of validation set =",sm)
write.csv(sm,paste0("E:\\AbhinavB\\Kaggle\\StoreItemDemandPrediction\\submission_files\\validationset_smape",format(Sys.time(),"%d-%b-%Y %H.%M"),".csv"))

predtest=predict(xgb_mod1,newdata = dtest)

sub=data.frame("id"=save_ids,"sales"=predtest)

write.csv(sub,"E:\\AbhinavB\\Kaggle\\StoreItemDemandPrediction\\submission_files\\sub1.csv",row.names =FALSE )
##fwrite(sub,"sub1.csv")

endtime=Sys.time()
timetaken=endtime-starttime
timetaken





























