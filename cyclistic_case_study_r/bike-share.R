# Load packages 
library(tidyverse)
library(tidyr)
library(purrr)
library(dplyr)
library(lubridate)
library(readr)
library(janitor)
library(scales)
library(leaflet)


# Load data 
df1 <- read_csv('Downloads/cyclistic_case_study/202104-divvy-tripdata.csv')
df2 <- read_csv('Downloads/cyclistic_case_study/202105-divvy-tripdata.csv')
df3 <- read_csv('Downloads/cyclistic_case_study/202106-divvy-tripdata.csv')
df4 <- read_csv('Downloads/cyclistic_case_study/202107-divvy-tripdata.csv')
df5 <- read_csv('Downloads/cyclistic_case_study/202108-divvy-tripdata.csv')
df6 <- read_csv('Downloads/cyclistic_case_study/202109-divvy-tripdata.csv')
df7 <- read_csv('Downloads/cyclistic_case_study/202110-divvy-tripdata.csv')
df8 <- read_csv('Downloads/cyclistic_case_study/202111-divvy-tripdata.csv')
df9 <- read_csv('Downloads/cyclistic_case_study/202112-divvy-tripdata.csv')
df10 <- read_csv('Downloads/cyclistic_case_study/202201-divvy-tripdata.csv')
df11 <- read_csv('Downloads/cyclistic_case_study/202202-divvy-tripdata.csv')
df12 <- read_csv('Downloads/cyclistic_case_study/202203-divvy-tripdata.csv')

# Concatenate in one df
df <- rbind(df1,df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12)
head(df)

# Lets have a quick look at our df
glimpse(df)

# Check for missing values
missing_values <- colSums(is.na(df))

# Print the columns with missing values
print(missing_values[missing_values > 0])

# Check for duplicated rows
duplicated_rows <- df[duplicated(df), ]

# Print the duplicated rows
print(duplicated_rows)

# Check for duplicates in a ride_id column 
duplicated_rows_id <- df[duplicated(df$ride_id), ]

# Print the duplicated rows
print(duplicated_rows_id)

# Remove missing values
df_clean <- drop_na(df, start_station_name)
df_clean <- drop_na(df_clean, end_station_name)

#Check for missing values again
missing_values2 <- colSums(is.na(df_clean))

# Print the columns with missing values
print(missing_values2[missing_values2 > 0])

# Lets check columns names
colnames(df_clean)

# Rename column
df_clean <- df_clean %>% 
  rename(customer_type = member_casual)

#Display structure of df
str(df_clean)

# Create new column
df_clean$ride_length <- difftime(
  df_clean$ended_at, 
  df_clean$started_at,
  units = "min"
) 

# Change datatype and round to 2 decimal points
df_clean$ride_length <- round(as.numeric(df_clean$ride_length), 2)

# Remove short trips
df_clean <- df_clean %>% 
  filter(df_clean$ride_length >1)

# New column with weekday
df_clean$week_day <- lubridate::wday(df_clean$started_at, label=TRUE)

#Month column
df_clean$month <- month(df_clean$started_at, label=TRUE)
head(df_clean)

# Check data types and column names one more time
glimpse(df_clean)

# Count number of rides for casual riders and member riders in particular.
customer_counts <- table(df_clean$customer_type)
customer_counts

# Visualize it as a pie chart
total_rides <- sum(customer_counts)
pie_percent <- round((customer_counts / total_rides) * 100, 2)
pie_label <- c("casual", "member")
pie_label <- paste(pie_label, "\n", pie_percent, "%")

pie(customer_counts, pie_percent,
    main = "Total number of rides \n April 2021 - March 2022",
    col= c("salmon", "darkturquoise"),
    labels = pie_label,
    cex=1.3)

#Calculate mean, max and min ride length 
mean(df_clean$ride_length)
max(df_clean$ride_length)
min(df_clean$ride_length)

# Calculate mean ride length for different customer type
mean_type <- aggregate(ride_length ~ customer_type, data = df_clean, FUN = mean)
mean_type

# Calculate mean right length for customer type and week day
mean_type_week <- aggregate(ride_length ~ customer_type + week_day, data = df_clean, FUN = mean)
mean_type_week

# Visualize average ride length per day of the week
ggplot(data = mean_type_week) +
  geom_col(mapping = aes(x=week_day, y=ride_length, fill=customer_type)) +
  facet_wrap(~customer_type) +
  theme(axis.text.x=element_text(angle = 45, hjust=1)) +
  labs(title = "Average ride length per day of the week")

# Count number of rides per weekday and customer type
week_day_rides <- df_clean %>%
  group_by(customer_type, week_day) %>%
  summarise(number_of_rides = n()) %>% 
  arrange(week_day)

week_day_rides

# Visualize numbers of rides per day of the week
ggplot(data = week_day_rides) +
  geom_col(mapping = aes(x=week_day, y=number_of_rides, fill=customer_type)) +
  facet_wrap(~customer_type) +
  theme(axis.text.x=element_text(angle = 45, hjust=1)) +
  labs(title = "Total numbers of rides per day of the week") +
  scale_y_continuous(labels = label_number(suffix = " K", scale = 1e-3))

# Number of rides per hours of the day and customer type
df_clean$hour <- format(df_clean$started_at, format = "%H")

hour_rides <- df_clean %>%
  group_by(customer_type, hour) %>%
  summarise(number_of_rides = n()) %>% 
  arrange(hour)
hour_rides

# Prepare visualization
ggplot(hour_rides) +
  geom_bar(mapping = aes(fill=customer_type, y=number_of_rides, x=hour), 
           position="dodge", stat="identity") +
  labs(title = "Total numbers of rides per hour") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank()) +
  scale_y_continuous(labels = label_number(suffix = " K", scale = 1e-3))

# Find number of rides per month 
month_rides <- df_clean %>% 
  group_by(customer_type,month) %>%  
  summarise(number_of_rides = n()) %>% 
  arrange(month) 
month_rides

# Distribution number of rides per month of the year
ggplot(month_rides, aes(fill=customer_type, y=number_of_rides, x=month)) + 
  geom_bar(position="dodge", stat="identity") +
  labs(title = "Total numbers of rides per month") +
  theme(axis.title.x = element_blank(), axis.title.y = element_blank()) +
  scale_y_continuous(labels = label_number(suffix = " K", scale = 1e-3))

# Count number of rides per rideable type
rideable_type_rides <- df_clean %>%
  group_by(customer_type, rideable_type) %>%
  summarise(number_of_rides = n()) %>% 
  arrange(rideable_type)
rideable_type_rides

# Visualize numbers of rides per rideable type
ggplot(data = rideable_type_rides) +
  geom_col(mapping = aes(x=rideable_type, y=number_of_rides, fill=customer_type)) +
  facet_wrap(~customer_type) +
  theme(axis.text.x=element_text(angle = 10), 
        axis.title.x = element_blank(), axis.title.y = element_blank()) +
  labs(title = "Total numbers of rides per rideable type") +
  scale_y_continuous(labels = label_number(suffix = " M", scale = 1e-6))


# Popular station for casual customers
station_casual <- df_clean %>% 
  group_by(start_station_name, customer_type) %>% 
  filter(customer_type == "casual") %>% 
  summarise(number_of_rides = n()) %>% 
  arrange(desc(number_of_rides))

head(station_casual, n=10)

# Popular station for member customers
station_member <- df_clean %>% 
  group_by(start_station_name, customer_type) %>% 
  filter(customer_type == "member") %>% 
  summarise(number_of_rides = n()) %>% 
  arrange(desc(number_of_rides)) 

head(station_member, n=10)

# Number of rides per station for both customer type
station_all_type <- df_clean %>% 
  group_by(start_station_name, customer_type) %>% 
  summarise(number_of_rides = n()) %>% 
  arrange(desc(number_of_rides)) 

# Data frame with coordinates
station_coord <- df_clean %>%
  select(start_station_name, start_lat, start_lng) %>%
  distinct(start_station_name, .keep_all = TRUE)

# Merge data
merged_data <- station_all_type %>%
  left_join(station_coord, by = "start_station_name")

head(merged_data)

# Create a Leaflet map object
map <- leaflet() %>%
  addTiles()

map <- map %>%
  addCircleMarkers(
    data = subset(merged_data[1:100,], customer_type == "casual"),
    lat = ~start_lat,
    lng = ~start_lng,
    label = ~paste(start_station_name, "Rides:", number_of_rides),
    color = "red",
    radius = 5,
    group = "Casual"
  ) %>%
  addCircleMarkers(
    data = subset(merged_data[1:100,], customer_type == "member"),
    lat = ~start_lat,
    lng = ~start_lng,
    label = ~paste(start_station_name, "Rides:", number_of_rides),
    color = "blue",
    radius = 5,
    group = "Member"
  )

map <- map %>%
  addLayersControl(
    overlayGroups = c("Casual", "Member"),
    options = layersControlOptions(collapsed = FALSE)
  )

map
