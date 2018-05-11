library(data.table)
library(dplyr)
library(lubridate)
library(ggplot2)
library(caret)
library(devtools)
library(lightgbm)
library(parallelMap)
library(parallel)

parallelStartSocket(cpus = detectCores())

train_data <- fread("../input/train.csv", sep = ",", header = TRUE)

train_data <- as.data.table(train_data)
train_data <- train_data[, -7] # attributed_time
train1 <- train_data[1:45000000,]
train2 <- train_data[45000001:90000000,]
train3 <- train_data[90000001:135000000,]
train4 <- train_data[135000001:184903890,]
rm(train_data)

train1[, click_time := ymd_hms(click_time) + hours(8)]
train2[, click_time := ymd_hms(click_time) + hours(8)]
train3[, click_time := ymd_hms(click_time) + hours(8)]
train4[, click_time := ymd_hms(click_time) + hours(8)]

train1[, `:=` (day, day(click_time))]
train1[, `:=` (hour, hour(click_time))]
train1 <- train1[, -c("click_time"), with = F]
train2[, `:=` (day, day(click_time))]
train2[, `:=` (hour, hour(click_time))]
train2 <- train2[, -c("click_time"), with = F]
train3[, `:=` (day, day(click_time))]
train3[, `:=` (hour, hour(click_time))]
train3 <- train3[, -c("click_time"), with = F]
train4[, `:=` (day, day(click_time))]
train4[, `:=` (hour, hour(click_time))]
train4 <- train4[, -c("click_time"), with = F]
invisible(gc())

train_day7 <- train1[day == 7]
rm(train1)
train_day7 <- rbind(train_day7, train2[day == 7])
train_day8 <- train2[day == 8]
rm(train2)
train_day8 <- rbind(train_day8, train3[day == 8])
train_day9 <- train3[day == 9]
rm(train3)
train_day9 <- rbind(train_day9, train4[day == 9])
rm(train4)
invisible(gc())

write.csv(train_day7, "train_day7.csv")
write.csv(train_day8, "train_day8.csv")
write.csv(train_day9, "train_day9.csv")

train_day7 <- fread("train_day7.csv", sep = ",", header = TRUE)
train_day8 <- fread("train_day8.csv", sep = ",", header = TRUE)
train_day9 <- fread("train_day9.csv", sep = ",", header = TRUE)

most_freq_hours_in_test_data <- c("12","13","17","18","21","22")
least_freq_hours_in_test_data <- c("14","19","23")

train_day7 <- train_day7 %>%
  mutate(in_test_hh = ifelse(hour %in% most_freq_hours_in_test_data, 1,
                             ifelse(hour %in% least_freq_hours_in_test_data, 3, 2))) %>%
  add_count(ip, day, in_test_hh) %>% rename("nip_day_test_hh" = n) %>%
  select(-c(in_test_hh)) %>%
  add_count(ip, day, hour) %>% rename("n_ip" = n) %>%
  add_count(ip, day, hour, os) %>% rename("n_ip_os" = n) %>% 
  add_count(ip, day, hour, app) %>% rename("n_ip_app" = n) %>%
  add_count(ip, day, hour, app, os) %>% rename("n_ip_app_os" = n) %>% 
  add_count(app, day, hour) %>% rename("n_app" = n) %>%
  select(-c(day,ip))

invisible(gc())

train_day8 <- train_day8 %>%
  mutate(in_test_hh = ifelse(hour %in% most_freq_hours_in_test_data, 1,
                             ifelse(hour %in% least_freq_hours_in_test_data, 3, 2))) %>%
  add_count(ip, day, in_test_hh) %>% rename("nip_day_test_hh" = n) %>%
  select(-c(in_test_hh)) %>%
  add_count(ip, day, hour) %>% rename("n_ip" = n) %>%
  add_count(ip, day, hour, os) %>% rename("n_ip_os" = n) %>% 
  add_count(ip, day, hour, app) %>% rename("n_ip_app" = n) %>%
  add_count(ip, day, hour, app, os) %>% rename("n_ip_app_os" = n) %>% 
  add_count(app, day, hour) %>% rename("n_app" = n) %>%
  select(-c(day,ip))

invisible(gc())

train_day9 <- train_day9 %>%
  mutate(in_test_hh = ifelse(hour %in% most_freq_hours_in_test_data, 1,
                             ifelse(hour %in% least_freq_hours_in_test_data, 3, 2))) %>%
  add_count(ip, day, in_test_hh) %>% rename("nip_day_test_hh" = n) %>%
  select(-c(in_test_hh)) %>%
  add_count(ip, day, hour) %>% rename("n_ip" = n) %>%
  add_count(ip, day, hour, os) %>% rename("n_ip_os" = n) %>% 
  add_count(ip, day, hour, app) %>% rename("n_ip_app" = n) %>%
  add_count(ip, day, hour, app, os) %>% rename("n_ip_app_os" = n) %>% 
  add_count(app, day, hour) %>% rename("n_app" = n) %>%
  select(-c(day,ip))

invisible(gc())

################### train_day7 ###################

train_index <- createDataPartition(train_day7$is_attributed, p = 0.9, list = FALSE)

day7_train <- train_day7[train_index, ]
day7_valid <- train_day7[-train_index, ]
rm(train_day7)

################### train_day8 ###################

train_index <- createDataPartition(train_day8$is_attributed, p = 0.9, list = FALSE)

day8_train <- train_day8[train_index, ]
day8_valid <- train_day8[-train_index, ]
rm(train_day8)

################### train_day9 ###################

train_index <- createDataPartition(train_day9$is_attributed, p = 0.9, list = FALSE)

day9_train <- train_day9[train_index, ]
day9_valid <- train_day9[-train_index, ]
rm(train_day9)

invisible(gc())

day7_train <- fread("day7_train.csv", sep = ",", header = TRUE)
day7_valid <- fread("day7_valid.csv", sep = ",", header = TRUE)
day8_train <- fread("day8_train.csv", sep = ",", header = TRUE)
day8_valid <- fread("day8_valid.csv", sep = ",", header = TRUE)
day9_train <- fread("day9_train.csv", sep = ",", header = TRUE)
day9_valid <- fread("day9_valid.csv", sep = ",", header = TRUE)


day9_train <- day9_train[,-c(1,2)]
day9_valid <- day9_valid[,-c(1,2)]

day9_train <- day9_train[sample(1:dim(day9_train)[1], 4000000 , replace=F) ,  ]
day9_valid <- day9_train[sample(1:dim(day9_train)[1], 400000 , replace=F) ,  ]

day9_train <- as.data.frame(day9_train)
day9_valid <- as.data.frame(day9_valid)

categorical_features = c("app", "device", "os", "channel", "hour")

## Creating lgb.Dataset (train, validation)
# train 
dtrain = lgb.Dataset(data = as.matrix(day9_train[, colnames(day9_train) != "is_attributed"]), 
                     label = day9_train$is_attributed, categorical_feature = categorical_features)

# validation
dvalid = lgb.Dataset(data = as.matrix(day9_valid[, colnames(day9_valid) != "is_attributed"]), 
                     label = day9_valid$is_attributed, categorical_feature = categorical_features)

invisible(gc())

params = list(objective = "binary", 
              metric = "auc", 
              learning_rate= 0.1,
              num_leaves= 7,
              max_depth= 3,
              min_child_samples= 100,
              max_bin= 100,
              subsample= 0.7, 
              subsample_freq= 1,
              colsample_bytree= 0.7,
              min_child_weight= 0,
              min_split_gain= 0,
              scale_pos_weight= 99.7)

# Modelling
model1 <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                    nrounds = 1000, verbose= 1, early_stopping_rounds = 30, eval_freq = 25)

rm(dtrain, dvalid)
invisible(gc())

cat("Validation AUC @ best iter: ", max(unlist(model1$record_evals[["validation"]][["auc"]][["eval"]])), "\n\n")


invisible(gc())

test_data <- fread("test_data.csv", sep = ",", header = TRUE)
test_data <- as.data.table(test_data)
test_data[, click_time := ymd_hms(click_time) + hours(8)]
test_data[, `:=` (day, day(click_time))]
test_data[, `:=` (hour, hour(click_time))]
test_data <- test_data[, -c("click_time"), with = F]
test_data1 <- test_data %>%
  mutate(in_test_hh = ifelse(hour %in% most_freq_hours_in_test_data, 1,
                             ifelse(hour %in% least_freq_hours_in_test_data, 3, 2))) %>%
  add_count(ip, day, in_test_hh) %>% rename("nip_day_test_hh" = n) %>%
  select(-c(in_test_hh)) %>%
  add_count(ip, day, hour) %>% rename("n_ip" = n) %>%
  add_count(ip, day, hour, os) %>% rename("n_ip_os" = n) %>% 
  add_count(ip, day, hour, app) %>% rename("n_ip_app" = n) %>%
  add_count(ip, day, hour, app, os) %>% rename("n_ip_app_os" = n) %>% 
  add_count(app, day, hour) %>% rename("n_app" = n) %>%
  select(-c(day,ip))

invisible(gc())

preds <- predict(model1, data = as.matrix(test_data1[, colnames(test_data1)], n = model1$best_iter))
preds <- as.data.frame(preds)

sub <- data.table(click_id = test_data$click_id, is_attributed = NA) 
sub$is_attributed <- preds

fwrite(sub, "submission.csv")
