# Fashion-retail-sales-prediction-using-R
This project analyzes customer purchasing and review behavior in a fashion retail dataset
This project analyzes customer purchasing and review behavior in a fashion retail dataset. It involves:

- Predicting review ratings using Random Forest regression.
- Forecasting future sales using ARIMA time series models.
- Segmenting customers using K-Means clustering.

The entire analysis is done using **R programming language** and popular data science libraries.

##Tools and Libraries Used

- **Data Wrangling**: `dplyr`, `tidyr`, `lubridate`
- **Visualization**: `ggplot2`, `factoextra`
- **Machine Learning**: `caret`, `randomForest`, `e1071`
- **Time Series**: `forecast`, `tsibble`, `ARIMA`
- **Clustering**: `cluster`, `kmeans`

---

## 📊 Dataset Description

**File:** `Fashion_Retail_Sales.csv`

| Column                 | Description                         |
|------------------------|-------------------------------------|
| `Customer.Reference.ID`| Unique customer ID                  |
| `Item.Purchased`       | Product bought                      |
| `Purchase.Amount..USD.`| Purchase amount in USD              |
| `Date.Purchase`        | Purchase date (format: dd-mm-yyyy) |
| `Review.Rating`        | Customer review rating              |
| `Payment.Method`       | Cash or Credit Card                 |

---

## 🧪 Project Breakdown

### 🔹 Step 1: Load Libraries and Dataset

```r
library(dplyr)
library(ggplot2)
library(lubridate)
library(tidyr)
library(caret)
library(randomForest)
library(e1071)
library(forecast)
library(cluster)
library(factoextra)
r
Copy
Edit
data <- read.csv("Fashion_Retail_Sales.csv", stringsAsFactors = FALSE)
View(data)
str(data)
🔹 Step 2: Clean and Prepare the Data
r
Copy
Edit
# Check for missing values
sapply(data, function(x) sum(is.na(x)))

# Remove rows with missing values
data <- na.omit(data)
r
Copy
Edit
# Rename columns
colnames(data) <- c("CustomerID", "ItemPurchased", "PurchaseAmountUSD", 
                    "DatePurchase", "ReviewRating", "PaymentMethod")

# Convert date format
data$DatePurchase <- as.Date(data$DatePurchase, format = "%d-%m-%Y")

# Extract month and day of week
data$Month <- months(data$DatePurchase)
data$DayOfWeek <- weekdays(data$DatePurchase)
🔹 Step 3: Predict Review Ratings Using Random Forest
r
Copy
Edit
# Split into training and test sets
set.seed(123)
train_index <- createDataPartition(data$ReviewRating, p = 0.8, list = FALSE)
train_reg <- data[train_index, ]
test_reg <- data[-train_index, ]
r
Copy
Edit
# Train Random Forest model
rf_reg_model <- randomForest(
  ReviewRating ~ ItemPurchased + PurchaseAmountUSD + Month + DayOfWeek,
  data = train_reg,
  ntree = 100,
  importance = TRUE
)
r
Copy
Edit
# Predict and evaluate
predictions_reg <- predict(rf_reg_model, newdata = test_reg)
postResample(predictions_reg, test_reg$ReviewRating)

# Attach predictions
test_reg$Predicted_rating <- predictions_reg
View(test_reg)
🔹 Step 4: Time Series Forecasting with ARIMA
r
Copy
Edit
# Convert numeric and date columns
data$PurchaseAmountUSD <- as.numeric(data$PurchaseAmountUSD)
data$DatePurchase <- as.Date(data$DatePurchase, format = "%d-%m-%Y")
r
Copy
Edit
# Aggregate daily sales
daily_sales <- data %>%
  group_by(DatePurchase) %>%
  summarise(TotalSales = sum(PurchaseAmountUSD, na.rm = TRUE))
r
Copy
Edit
# Fit ARIMA model
fit_arima <- auto.arima(daily_sales$TotalSales)

# Forecast next 30 days
forecast_sales <- forecast(fit_arima, h = 30)
r
Copy
Edit
# Plot forecast
autoplot(forecast_sales) +
  ggtitle("30-Day Sales Forecast") +
  xlab("Date") +
  ylab("Sales Amount")
🔹 Step 5: Customer Segmentation Using K-Means Clustering
r
Copy
Edit
# Select relevant features
cluster_data <- data %>%
  select(PurchaseAmountUSD, ReviewRating) %>%
  na.omit()

# Scale features
cluster_scaled <- scale(cluster_data)
r
Copy
Edit
# Find optimal clusters using elbow method
fviz_nbclust(cluster_scaled, kmeans, method = "wss") +
  labs(title = "Elbow Method to Determine Optimal k")
r
Copy
Edit
# Apply K-Means clustering (3 clusters)
kmeans_model <- kmeans(cluster_scaled, centers = 3, nstart = 25)
r
Copy
Edit
# Add cluster labels to original data
clustered_data <- data[match(rownames(cluster_data), rownames(data)), , drop = FALSE]
clustered_data$Cluster <- as.factor(kmeans_model$cluster)
r
Copy
Edit
# Visualize clusters
fviz_cluster(kmeans_model, data = cluster_scaled, 
             palette = c("#E7B800", "#4E79A7", "#A0C4E8"),
             main = "Clustering of Products by Purchase Amount & Review Rating")
📈 Results Summary
📌 Review Rating Prediction
Model: Random Forest Regression

Metrics:

RMSE ≈ 1.15

R-squared ≈ 0.002 (model not very accurate)

MAE ≈ 0.99

📌 Sales Forecasting
Model: ARIMA

Forecast: Flat sales forecast of ≈ 1180.69 USD per day

Note: No trend or seasonality detected in ARIMA(0,0,0)

📌 Clustering
Method: K-Means with 3 clusters

Insights:

Cluster 1: Low spenders, low reviews

Cluster 2: Mid spenders, mixed reviews

Cluster 3: High spenders, high reviews
## 🧩 Project Performance Review

While this project successfully demonstrates a full machine learning workflow on fashion retail sales data, the **predictive performance was relatively poor**, especially for the Random Forest regression model used to predict review ratings.

### ❌ Reasons for Poor Model Performance

1. **Lack of Strong Predictive Features**  
   The features used (e.g., `ItemPurchased`, `PurchaseAmountUSD`, `Month`, `DayOfWeek`) may not be sufficient to explain customer review behavior. Customer satisfaction is complex and may depend on factors not included in the dataset, such as delivery experience, product quality, or customer service.

2. **No Textual or Sentiment Data**  
   If customer comments or feedback were available, they could significantly improve prediction accuracy using NLP techniques.

3. **Categorical Variable Handling**  
   The variable `ItemPurchased` was likely treated as a high-cardinality categorical feature, which can be noisy and reduce model generalizability unless encoded properly.

4. **Flat ARIMA Forecast**  
   The time series model used (ARIMA(0,0,0)) suggests no discernible trend or seasonality in the sales data. This may be due to limited variability or insufficient time span in the dataset.

5. **Data Quality Issues**  
   Around 27% of the dataset was removed due to missing values. This could have led to a loss of valuable information and reduced model robustness.

##  **Recommendations
1. **Enhance Dataset with More Features**  
   Include features such as:
   - Product categories or brand
   - Delivery time or status
   - Customer demographics
   - Previous purchase history
   - Loyalty or membership status

2. **Feature Engineering**  
   Create interaction features like:
   - Revenue per product
   - Frequency of purchases
   - Time since last purchase

3. **Improve Data Collection and Cleaning**  
   Minimize missing values through better data logging or imputation. Retain as much information as possible without sacrificing quality.

4. **Apply Better Encoding Techniques**  
   For high-cardinality categorical variables like `ItemPurchased`, use:
   - Target encoding
   - Embedding layers (if using deep learning)
   - PCA on one-hot encoded features

5. **Explore Other Models**  
   - Use ensemble methods like Gradient Boosting Machines (GBM), XGBoost, or LightGBM.
   - For time series, explore Prophet or STL decomposition.
   - Consider regression trees with quantile regression for better interval estimation.

6. **Collect Textual Feedback**  
   If available, apply sentiment analysis or topic modeling to customer reviews to gain deeper insights and improve model inputs.

By implementing the above suggestions, future iterations of the model will likely achieve better predictive performance and yield more actionable business insights.


