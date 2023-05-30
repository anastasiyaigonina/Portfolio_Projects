-- How many properties are listed for sale by state
SELECT state,
COUNT(*) AS properties_count 
FROM [realtor].[dbo].[realtor-data]
WHERE status = 'for_sale'
GROUP BY state 
ORDER BY properties_count DESC;

-- Average number of bedrooms and bathrooms for properties that is ready to sale 
SELECT AVG(bed) AS avg_bedrooms,
AVG(bath) AS avg_bathrooms
FROM [realtor].[dbo].[realtor-data]
WHERE status = 'for_sale';

-- Average, min and max prices by state
SELECT state, 
ROUND(AVG(price),0) AS avg_price, 
MIN(price) AS min_price,
MAX(price) AS max_price
FROM [realtor].[dbo].[realtor-data]
GROUP BY state
ORDER BY state;

-- Median price by state
SELECT DISTINCT state, percentile_cont(0.5) WITHIN GROUP (ORDER BY price) OVER (PARTITION BY state) AS median_price
FROM [realtor].[dbo].[realtor-data]
ORDER BY state;

-- Average price for square feet for different states
SELECT state,
ROUND(AVG(price / house_size),0) AS avg_price_per_sqft 
FROM [realtor].[dbo].[realtor-data]
WHERE house_size IS NOT NULL
GROUP BY state 
ORDER BY state;

-- Top 10 most expensive cities by average price
SELECT TOP(10) city, ROUND(AVG(price),0) AS avg_price
FROM [realtor].[dbo].[realtor-data]
GROUP BY city
ORDER BY avg_price DESC;

-- List of properties that is more expensive than average price in this city
WITH city_avg_price AS (
    SELECT city, ROUND(AVG(price),0) AS avg_price
    FROM [realtor].[dbo].[realtor-data]
    GROUP BY city
)
SELECT rd.*,
       cap.avg_price AS city_average_price
FROM [realtor].[dbo].[realtor-data]rd
JOIN city_avg_price cap ON rd.city = cap.city
WHERE rd.price > cap.avg_price;

-- Find mean, min and max houses sizes by state
SELECT state, 
ROUND(AVG(house_size),0) AS avg_size, 
MIN(house_size) AS min_size,
MAX(house_size) AS max_size
FROM [realtor].[dbo].[realtor-data]
WHERE house_size IS NOT NULL
GROUP BY state
ORDER BY state;

-- Properties with latest sold date for each city
WITH earliest_sold_properties AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY city ORDER BY prev_sold_date DESC) AS property_rank
    FROM [realtor].[dbo].[realtor-data]
    WHERE prev_sold_date IS NOT NULL
)
SELECT *
FROM earliest_sold_properties
WHERE property_rank = 1
ORDER BY city, prev_sold_date DESC;

-- Create categories for bedrooms and count amount of properties in each category
SELECT bedroom_category, COUNT(*) AS property_count
FROM (
    SELECT *,
           CASE
               WHEN bed BETWEEN 1 AND 2 THEN 'small'
               WHEN bed BETWEEN 2 AND 3 THEN 'medium'
               ELSE 'large'
           END AS bedroom_category
    FROM [realtor].[dbo].[realtor-data]
) subquery
GROUP BY bedroom_category;

-- Percentage of expensive properties for each number of bathrooms
SELECT bath,
       COUNT(*) AS property_count,
       AVG(CASE WHEN price > 500000 THEN 1.0 ELSE 0.0 END) * 100 AS percentage_expensive_properties
FROM [realtor].[dbo].[realtor-data]
WHERE bath IS NOT NULL
GROUP BY bath
ORDER BY bath;

-- How many properties have a house size within one standard deviation of the average house size in their respective state?
SELECT state, COUNT(*) AS properties_count
FROM (
    SELECT *,
           AVG(house_size) OVER (PARTITION BY city) AS avg_city_house_size,
           STDEV(house_size) OVER (PARTITION BY city) AS stddev_city_house_size
    FROM [realtor].[dbo].[realtor-data]
) subquery
WHERE house_size BETWEEN avg_city_house_size - stddev_city_house_size AND avg_city_house_size + stddev_city_house_size
GROUP BY state;

-- What is the cumulative sum of the house sizes for properties in each combination of city and state, ordered by the cumulative sum in descending order?
SELECT DISTINCT city, state, 
       SUM(house_size) OVER (PARTITION BY city, state) AS cumulative_sum
FROM [realtor].[dbo].[realtor-data]
WHERE house_size IS NOT NULL
ORDER BY cumulative_sum DESC;

-- Calculate the price difference between consecutive sales within each state, excluding duplicated rows
SELECT state, prev_sold_date, price,
           price - LAG(price) OVER (PARTITION BY state ORDER BY prev_sold_date) AS price_difference
FROM (
    SELECT state, prev_sold_date, price,
    ROW_NUMBER() OVER (PARTITION BY state, price, prev_sold_date ORDER BY price) AS row_num
    FROM [realtor].[dbo].[realtor-data]
    WHERE prev_sold_date IS NOT NULL
    ) sub
WHERE row_num = 1
ORDER BY state, prev_sold_date;
