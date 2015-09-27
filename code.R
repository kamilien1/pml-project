# project code



# open up the file in excel and we find
# multiple NA types
raw_df <- read.csv("data//pml-training.csv")
unique(raw_df$kurtosis_picth_belt)[1:2]
# lots of factors
str(raw_df)

# adjust for these na types
train <- read.csv("data//pml-training.csv", na.strings=c("NA","#DIV/0!",""))
test <- read.csv("data//pml-testing.csv", na.strings=c("NA","#DIV/0!",""))

# now we have factors replaces with num and lots of NA vals
str(train)


# for reproducible research, set seed 
set.seed(1)


library(caret,verbose=F)
inTrain <- createDataPartition(y=train$classe,
                               p=0.75, list=FALSE)
training <- train[inTrain,]
testing <- train[-inTrain,]
library(dplyr)
rows <- dim(training)[1] + dim(testing)[1]
training %>% rbind(testing) %>% select(classe) %>% 
    group_by(classe) %>% summarise(ratio=round(n()/rows,2))



near_zero_var <- nearZeroVar(training, saveMetrics=TRUE)
post_nzv_training <- training[,(!near_zero_var$nzv)]
names(post_nzv_training)[1:6]
post_nzv_training <- post_nzv_training[,-(1:6)]
post_nzv_testing <- testing[,(!near_zero_var$nzv)]
post_nzv_testing <- post_nzv_testing[,-(1:6)]


# get rid of high na columns
high_na <- apply(post_nzv_training, 2, function(x) 
                sum(is.na(x)))/nrow(post_nzv_training)
post_nzv_na_training <- post_nzv_training[!(high_na > 0.9)]
dim(post_nzv_na_training)
dep_var <- which(names(post_nzv_na_training)=='classe')
high_na_test <- apply(post_nzv_testing, 2, function(x) 
    sum(is.na(x)))/nrow(post_nzv_training)
post_nzv_na_testing <- post_nzv_testing[!(high_na > 0.9)]


# now do PCA
preProc <- preProcess(post_nzv_na_training[,-c(dep_var)],method='pca',thresh=0.95)
preProc
post_nzv_na_trainingPC <- predict(preProc,post_nzv_na_training[,-c(dep_var)])
post_nzv_na_testingPC <- predict(preProc,post_nzv_na_testing[,-c(dep_var)])


# random forests
library(randomForest)
modelFit <- randomForest(training$classe ~ .,
                         data=post_nzv_na_trainingPC, do.trace=F,
                         importance=T)

# get the importance
# http://www.statistik.uni-dortmund.de/useR-2008/slides/Strobl+Zeileis.pdf
# types: 1=mean decrease in accuracy, 2=mean decrease in node impurity


# Here are the definitions of the variable importance measures. 
# The first measure is computed from permuting OOB data: For each tree, the 
# prediction error on the out-of-bag portion of the data is recorded 
# (error rate for classification, MSE for regression). Then the same is done 
# after permuting each predictor variable. The difference between the two are 
# then averaged over all trees, and normalized by the standard deviation of 
# the differences. If the standard deviation of the differences is equal to 
# 0 for a variable, the division is not done (but the average is almost 
# always equal to 0 in that case).

# The second measure is the total decrease in node impurities from splitting on 
# the variable, averaged over all trees. For classification, the node impurity is 
# measured by the Gini index. For regression, it is measured by residual sum of squares.

modelFit$importance
with(modelFit,plot(1:length(importance),importance))
cbind(importance(modelFit,1),importance(modelFit,2))
# now let's look at accuracy
confusionMatrix(testing$classe,predict(modelFit,post_nzv_na_testingPC))









