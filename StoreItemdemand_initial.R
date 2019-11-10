
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


##Notations off and clear all objects
options(scipen = 999)
rm(list = ls())
gc()

startime=Sys.time()

zerovari <- function(dat) {
  out <- lapply(dat, function(x) length(unique(x)))
  want <- which(!out > 1)
  unlist(want)
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

storedata_orig=fread("E:\\AbhinavB\\Kaggle\\StoreItemDemandPrediction\\train.csv",
                   data.table = FALSE,stringsAsFactors = TRUE)
                   

storedata=storedata_orig
dim(storedata)


