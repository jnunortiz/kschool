library(lattice)
library(plyr)
library(ggplot2)
library(reshape2)
library(e1071)
library(ranger)
library(caret)
library(Formula)
library(ipred)

set.seed(123)

flat <- dget('residential.sale.flat.pickle')
house <- dget('residential.sale.house.pickle')

flat$ISFLAT <- rep(1, nrow(flat))
house$ISFLAT <- rep(0, nrow(house))

rozas <- rbind(flat, house)

rozas$STATEID <- NULL
rozas$LEVEL8ID <- NULL
rozas$PROPERTYTYPEID <- NULL
rozas$NEWDEVELOPMENTID <- NULL
rozas$ISFROMBANKID <- NULL

rozas$AREARANGETK <- as.factor(rozas$AREARANGETK)
rozas$FLATLOCATIONID <- as.factor(rozas$FLATLOCATIONID)
rozas$ENERGYCERTIFICATIONID <- as.factor(rozas$ENERGYCERTIFICATIONID)
rozas$BUILTTYPEID <- as.factor(rozas$BUILTTYPEID)

dummies <- dummyVars(PRICE ~ ., data = rozas)
rozas_processed <- as.data.frame(predict(dummies, rozas))

rozas_processed$HASPARKINGSPACEID <- as.factor(rozas$HASPARKINGSPACEID)
rozas_processed$HASSWIMMINGPOOLID <- as.factor(rozas$HASSWIMMINGPOOLID)
rozas_processed$HASTERRACEID <- as.factor(rozas$HASTERRACEID)
rozas_processed$HASGARDENID <- as.factor(rozas$HASGARDENID)
rozas_processed$HASLIFTID <- as.factor(rozas$HASLIFTID)
rozas_processed$ISINVOICEID <- as.factor(rozas$ISINVOICEID)
rozas_processed$ISFLAT <- as.factor(rozas$ISFLAT)
rozas_processed$HASAIRCONDITIONINGID <- as.factor(rozas$HASAIRCONDITIONINGID)

cols <- c("AREARANGETK.1", "AREARANGETK.2", "AREARANGETK.3", "AREARANGETK.4", "AREARANGETK.5", "AREARANGETK.6" ,"AREARANGETK.7",
          "AREARANGETK.8", "AREARANGETK.9","AREARANGETK.10", "AREARANGETK.11", "AREARANGETK.12", "AREARANGETK.13", "AREARANGETK.14",
          "AREARANGETK.15", "ENERGYCERTIFICATIONID.-1", "ENERGYCERTIFICATIONID.1", "ENERGYCERTIFICATIONID.2",
          "ENERGYCERTIFICATIONID.3",  "ENERGYCERTIFICATIONID.4",  "ENERGYCERTIFICATIONID.5",  "ENERGYCERTIFICATIONID.6",
          "ENERGYCERTIFICATIONID.7", "ENERGYCERTIFICATIONID.8", "ENERGYCERTIFICATIONID.9", "ENERGYCERTIFICATIONID.10", "BUILTTYPEID.1",
          "BUILTTYPEID.2", "BUILTTYPEID.3", "FLATLOCATIONID.-1", "FLATLOCATIONID.1", "FLATLOCATIONID.2", "YEAR.2006", "YEAR.2007",
          "YEAR.2008", "YEAR.2009", "YEAR.2010", "YEAR.2011", "YEAR.2012", "YEAR.2013", "YEAR.2014", "YEAR.2015", 
          "YEAR.2016", "YEAR.2017")

rozas_processed[cols] <- lapply(rozas_processed[cols], factor)

rozas_processed$YEAR.2100 <- NULL

rozas_2 <- data.frame(rozas$PRICE, rozas_processed)

tbl <- table(rozas_2$ROOMSID, rozas_2$HASPARKINGSPACEID)
chisq.test(tbl)

tbl2 <- table(rozas_2$ROOMSID, rozas_2$HASLIFTID)
chisq.test(tbl2)

tbl3 <- table(rozas_2$HASPARKINGSPACEID, rozas_2$HASLIFTID)
chisq.test(tbl3)

summary(aov(rozas.PRICE ~ ROOMSID + HASPARKINGSPACEID + HASLIFTID, data = rozas_2))

# shuffle rows and disorder data randomly
rows <- sample(nrow(rozas_2))
rozas_shuffled <- rozas_2[rows,]

# determine 80/20 split indexes
split <- round(nrow(rozas_shuffled) * 0.80)


# set train and test sets
rozas_train <- rozas_shuffled[1:split,]
rozas_test <- rozas_shuffled[(split+1):nrow(rozas_shuffled),]

# Fit Linear model: model_lm
model_lm <- train(
  rozas.PRICE ~ ., rozas_train,
  method = "lm",
  preProcess = c('zv', 'center', 'scale'),
  trControl = trainControl(
    method = "cv", number = 5,
    repeats = 1000, verboseIter = TRUE
  )
)

# Fit Random Forest model: model_rf
model_rf <- train(
  rozas.PRICE ~ ., rozas_train,
  method = "ranger",
  tuneLength = 1000,
  tuneGrid = data.frame(mtry = 1:10),
  trControl = trainControl(
    method = "cv", number = 5,
    repeats = 1000, verboseIter = TRUE
  )
)


# Fit Decision Boosted Tree: model_bt
model_bt <- train(
  rozas.PRICE ~ ., rozas_train,
  tuneGrid = data.frame(mstop = 100, maxdepth = 10, nu = 0.0001),
  method = "bstTree",
  trControl = trainControl(
    method = "cv", number = 5,
    repeats = 1000, verboseIter = TRUE
  )
)


# make list of models
model_list <- list(lm = model_lm, rforest = model_rf, boostTree = model_bt)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)

# plot 

bwplot(resamples, metric = 'RMSE')
bwplot(resamples, metric = 'Rsquared')

test_data <- rozas_test[-1]

prediction_lm <- predict(model_lm, test_data)
errors_lm <- rozas_test$rozas.PRICE - prediction_lm
rmse_lm <- sqrt(mean(errors_lm^2))

prediction_bt <- predict(model_bt, test_data)
errors_bt <- rozas_test$rozas.PRICE - prediction_bt
rmse_bt <- sqrt(mean(errors_bt^2))

prediction_rf <- predict(model_rf, test_data)
errors_rf <- rozas_test$rozas.PRICE - prediction_rf
rmse_rf <- sqrt(mean(errors_rf^2))

rmse <- data.frame(rmse_lm = rmse_lm, rmse_bt = rmse_bt, rmse_rf = rmse_rf)
rmse

quantile(errors_lm, c(0.05, 0.95))
quantile(errors_rf, c(0.05, 0.95))
quantile(errors_bt, c(0.05, 0.95))

predictions <- data.frame(Price_per_m2 = rozas_test$rozas.PRICE/rozas_test$CONSTRUCTEDAREA,
                          linear_model_prediction = prediction_lm/rozas_test$CONSTRUCTEDAREA,
                          random_forest_prediction = prediction_rf/rozas_test$CONSTRUCTEDAREA,
                          boosting_prediction = prediction_bt/rozas_test$CONSTRUCTEDAREA,
                          index = 1:nrow(rozas_test))

toPlot <- melt(predictions, id.vars = c('Price_per_m2', 'index'))

ggplot() + 
  geom_line(data = subset(toPlot, index <= 100), aes(y = value, x = index, col = variable))+
  facet_grid(variable ~.) +
  geom_line(data = subset(toPlot, index <= 100), aes(y = Price_per_m2, x = index)) +
  theme(legend.position="none", plot.title = element_text(hjust = 0.5)) +
  ggtitle('REAL VS PREDICTED') + 
  ylab('Price/m2') + 
  xlab('#Entry')

predictions <- data.frame(Price_per_m2 = rozas_test$rozas.PRICE,
                         linear_model_prediction = prediction_lm,
                         random_forest_prediction = prediction_rf,
                         boosting_prediction = prediction_bt,
                         index = 1:nrow(rozas_test))

toPlot <- melt(predictions, id.vars = c('Price_per_m2', 'index'))

ggplot() + 
  geom_line(data = subset(toPlot, index <= 100), aes(y = value, x = index, col = variable))+
  facet_grid(variable ~.) +
  geom_line(data = subset(toPlot, index <= 100), aes(y = Price_per_m2, x = index)) +
  theme(legend.position="none", plot.title = element_text(hjust = 0.5)) +
  ggtitle('REAL VS PREDICTED') + 
  ylab('Price/m2') + 
  xlab('#Entry')
