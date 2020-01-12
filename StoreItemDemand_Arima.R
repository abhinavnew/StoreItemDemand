
library(caret)
library(tidyr)
library(plyr)
library(dplyr)
library(caTools)
library(reshape2)
library(gbm)
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
library(forecast)
library(Metrics)
library(zoo)


##Notations off and clear all objects
options(scipen = 999)
rm(list = ls())
gc()

starttime=Sys.time()

first_day_of_month_wday <- function(dt) {
  day(dt) <- 1
  wday(dt)
}


storedata_origtrain=fread("E:\\AbhinavB\\Kaggle\\StoreItemDemandPrediction\\train.csv",
                          data.table = FALSE,stringsAsFactors = TRUE)
#storedata_origtrain=read_csv("../input/demand-forecasting-kernels-only/train.csv")

storedata_origtest=fread("E:\\AbhinavB\\Kaggle\\StoreItemDemandPrediction\\test.csv",data.table = FALSE,stringsAsFactors = TRUE)
#storedata_origtest=read_csv("../input/test.csv")
dim(storedata_origtrain)
dim(storedata_origtest)
storedata_train=storedata_origtrain
storedata_test=storedata_origtest
storedata_train$id=NA
storedata_train$ind="train"
storedata_test$sales=NA
storedata_test$ind="test"
combined=rbind(storedata_train,storedata_test)
dim(combined)
glimpse(combined)
storedata=combined
dim(storedata)

##date type feature creation 

storedata$date2=ymd(storedata$date)
storedata$YEAR=lubridate::year(storedata$date2)
storedata$MNTH=lubridate::month(storedata$date2)
storedata$Quarter=lubridate::quarter(storedata$date2)
storedata$Semester=lubridate::semester(storedata$date2)
storedata$MNTH_NM=lubridate::month(storedata$date2,label=TRUE)
##Day of week month year 
storedata$DayOfWeek=lubridate::wday(storedata$date2)
storedata$DayOfWeek_Name=lubridate::wday(storedata$date2,label=TRUE)
storedata$DayOfMonth=lubridate::mday(storedata$date2)
storedata$DayOfYear=lubridate::yday(storedata$date2)
storedata$DayOfQuarter=lubridate::qday(storedata$date2)
##Week of year & month
storedata$WeekOfYr=lubridate::week(storedata$date2)
storedata$WeekOfMnth=ceiling((day(storedata$date2)+first_day_of_month_wday(storedata$date2)-1)/7)



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

##subset preparation

##store1item1=trainset %>% filter(trainset$store==1 & trainset$item==1)
##store1item1=store1item1[,-which(names(store1item1) %in% c("id","ind","date2","store","item","date"))]
trainset_ref=trainset[,-which(names(trainset) %in% c("id","ind","date2","store","item","date"))]
head(trainset_ref)
length(trainset_ref)
##store1item1_val=valset %>% filter(valset$store==1 & valset$item==1)
valset_ref=valset[,-which(names(valset) %in% c("id","ind","date2","store","item","date"))]
head(valset_ref)
length(valset_ref)


#create ts object 


inds=seq(from=as.Date("2013-01-01"),to=as.Date("2016-12-31"),by="day")   ## this provides clear cut x axis values year wise 
last=length(inds)
Firstdayofyear=as.numeric(format(inds[1], "%j"))
Lastdayofyear=as.numeric(format(inds[last],"%j"))
sales_ts=ts(trainset$sales,start = c(2013,Firstdayofyear),end=c(2016,Lastdayofyear),frequency = 365)


length(sales_ts)
str(sales_ts)
head(sales_ts)
tail(sales_ts)

#decompose in cyclical ,trend,seasonal and residuals components

train_ts_comp=decompose(sales_ts)
autoplot(train_ts_comp)

###CHECK ---NOT ABLE TO ANALYSE DATA but trending exists ---deeper analysis reqd

#ARIMA prep

acf(sales_ts,lag.max = 60)
##Acf plot reveals that autocorr is not cutting off indicating non stationary TS
pacf(sales_ts,lag.max = 60)
##pacf plot reveals that partial autocorr not cutting off either 
##acf graph doesnot cut of below significance level hence ts is non stationary 
##find after how many differences will the timeseries become stationary

ndiffs(sales_ts)
##results in 1

#first differencing 
sales_ts_stnry=diff(sales_ts,differences=1)
plot.ts(sales_ts_stnry)

acf(sales_ts_stnry)
pacf(sales_ts_stnry)

##p -7 lags from pacf AR process
## q -1 lags from acf MA process 
##d -1 from ndiffs and sationary TS 
##combinations (p,d,q) -(0,1,0);(0,0,1);(0,1,1);(7,0,0);(7,1,1)


## creating grid for p and q 
prange=seq(0,7,by=1)
qrange=seq(0,7,by=1)
d=1
best_aic=9999999
best_p=0
best_q=0

# ##loop for finding p,d,q with lowest AIC -WARNING it takes a lot of time
# for(i in prange)
# { 
#   for(j in qrange)
#      { 
#         arima_reg=arima(sales_ts,order=c(i,d,j))
#         akic=arima_reg$aic
#         print(i)
#         print(j)
#         print(akic)
#         if (akic<best_aic)
#         { 
#            best_aic=akic
#            best_p=i
#            best_q=j
#            bestmodel=arima_reg
#         }
#         
#         
#     }
# }

print("best aic")
print(best_aic)
print("best parameters")
print(best_p)
print(d)
print(best_q)

## for the data set the best params were -(7,1,7)
arima6=arima(sales_ts,order = c(7,1,7))
arima6$aic
##seasonal ARIMA 

sales_ts_seadiff=diff(sales_ts,lag=7,differences=1)
plot(sales_ts_seadiff, type="l", main="Seasonally Differenced Time Series")
ndiffs(sales_ts_seadiff)
acf(sales_ts_seadiff,lag=60) ##7 ##1
pacf(sales_ts_seadiff,lag=60) ##6 ##1

##sarima3=arima(sales_ts,order = c(7,1,7),seasonal = list(order=c(0,1,1),period=7))
##sarima3$aic


##AUTO ARIMA 

fit=auto.arima(sales_ts,trace = TRUE,allowmean = F,allowdrift = FALSE)
summary(fit)
##making prediction using best SARIMA model-chosen basis of lowest AIC value

valset90=valset_ref[1:90]
pred_autoarima=forecast(fit,h=90)
autoplot(pred_autoarima)
length(pred_autoarima)
head(pred_autoarima)
predicted=as.numeric(pred_autoarima$mean)
length(predicted)

sm=smape(valset90,predicted)
sm

sarima3$x=sales_ts
pred_sarima=forecast(sarima3,h=90)
autoplot(pred_sarima)
length(pred_sarima)
predicted=as.numeric(pred_sarima$mean)
length(predicted)
sm=smape(valset90,predicted)
sm
print("smape of validation set =",sm)
write.csv(sm,paste0("E:\\AbhinavB\\Kaggle\\StoreItemDemandPrediction\\submission_files\\validationset_smape",format(Sys.time(),"%d-%b-%Y %H.%M"),".csv"))


##Holtwinter exponential smoothing 

train_hwes=HoltWinters(sales_ts)
summary(train_hwes)
plot(train_hwes)
train_hwes_frcst=forecast:::forecast.HoltWinters(train_hwes,h=90)
forecast:::plot.forecast(train_hwes_frcst)
predicted=as.numeric(train_hwes_frcst$mean)
sm=smape(valset90,predicted)
sm


predtest=predict(xgb_mod1,newdata = dtest)

sub=data.frame("id"=save_ids,"sales"=predtest)

write.csv(sub,"E:\\AbhinavB\\Kaggle\\StoreItemDemandPrediction\\submission_files\\sub1.csv",row.names =FALSE )
##fwrite(sub,"sub1.csv")

endtime=Sys.time()
timetaken=endtime-starttime
timetaken





























