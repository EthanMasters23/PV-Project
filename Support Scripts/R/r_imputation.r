"
Last updated: on Wednesday Apr 18 16:03 2022

@author: Ethan Masters

Purpose: Data Imputation Script Using R Packages

R Version:
"

print('Starting Data imputation using R script')

# load, and shape the data frame

data_frame <- read.csv("R_df_.csv")

data_frame[['Timestamp']] <- as.POSIXct(data_frame$Timestamp,
                                   format = "%Y-%m-%d %H:%M:%S")

seconds_data <- c(data_frame['seconds'])

data_frame <- subset(data_frame, select = -c(seconds))

# initiate ouput data frames

columns <- colnames(data_frame)

forecast_int_df <- data.frame(matrix(nrow = length(data_frame$Timestamp), ncol = length(columns))) 
colnames(forecast_int_df) <- columns
forecast_int_df$Timestamp <- data_frame$Timestamp

interpolation_df <- data.frame(matrix(nrow = length(data_frame$Timestamp), ncol = length(columns))) 
colnames(interpolation_df) <- columns
interpolation_df$Timestamp <- data_frame$Timestamp

spline_df <- data.frame(matrix(nrow = length(data_frame$Timestamp), ncol = length(columns))) 
colnames(spline_df) <- columns
spline_df$Timestamp <- data_frame$Timestamp

# forecast_arima_df <- data.frame(matrix(nrow = length(data_frame$Timestamp), ncol = length(columns))) 
# colnames(forecast_arima_df) <- columns
# forecast_arima_df$Timestamp <- data_frame$Timestamp

library(zoo)
library(forecast)

# loop through columns and impute missing values

for (i in colnames(data_frame)){

    if(i=='Timestamp') next

    zoo_obj <- zoo(x = data_frame[i], order.by = data_frame[['Timestamp']])

    interpolation_zoo <- data.frame(na.approx(zoo_obj))

    spline_zoo <- data.frame(na.spline(zoo_obj))

    forecast_int_zoo <- data.frame(na.interp(zoo_obj))

    interpolation_df[i] <- interpolation_zoo[,1]

    forecast_int_df[i] <- forecast_int_zoo[,1]

    spline_df[i] <- spline_zoo[,1]


    

    # ts <- ts(data_frame[i], start = data_frame[['Timestamp']])
    
    # arima <- data.frame(na.StructTS(ts))

    # forecast_arima_df[i] <- arima[,1]

    # ####

    # arima <- auto.arima(zoo_obj,trace = TRUE)

    # plot(forecast(arima))

    # print(forecast(arima,h=data_frame[['Timestamp']]))

    # # forecast_arima_df[i] <- forecast(arima)[,1]

}

# Load readr package
library(readr)

# Write to CSV file
# print(interpolation_df)
write_csv(interpolation_df, "interpolation_df.csv")

# Write to CSV file
# print(spline_df)
write_csv(spline_df, "spline_df.csv")

# Write to CSV file
# print(forecast_int_df)
write_csv(forecast_int_df, "forecast_int_df.csv")

# Write to CSV file
# print(forecast_int_df)
# write_csv(forecast_arima_df, "forecast_arima.csv")

print('Finished Data Imputation using R script')