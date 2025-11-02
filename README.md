## Problem Statement

This comprehensive data analysis project explores the **Medical Cost Personal Dataset** to identify key factors driving healthcare expenses and build predictive models for insurance costs. The analysis helps insurance companies, healthcare providers, and policymakers understand cost drivers and develop **data-driven strategies** for risk assessment, premium calculation, and preventive healthcare initiatives.

By uncovering relationships between demographic factors, lifestyle choices, and medical charges, this project enables **informed decision-making** to optimize healthcare resource allocation and promote cost-effective insurance solutions.

## Project Resources
- Dataset: [insurance.csv](https://github.com/user-attachments/files/23290047/insurance.csv)


- Source: Medical Cost Personal Dataset from Kaggle
- Google Colab Notebook: [Project File](https://colab.research.google.com/drive/1Jap-pw2HJmKhp7a1xpma6We5oECmfxOw?usp=sharing)


## Steps Followed

### 1. Data Loading & Inspection
* Imported the `insurance.csv` dataset using pandas
* Displayed first and last 5 rows for data preview
* Analyzed dataset shape, column names, and data types
* Generated summary statistics for numerical columns
* Identified missing values and duplicate records
* Created comprehensive markdown summary of initial findings

### 2. Data Cleaning & Preprocessing
* Removed duplicate rows (1 duplicate identified and removed)
* Standardized categorical columns (`sex`, `smoker`, `region`) to lowercase
* Validated data types and handled missing values (none found)
* Ensured dataset consistency and accuracy
* Documented cleaning outcomes with markdown explanations

### 3. Feature Engineering
* Created `Age_Group` column with categories:
  - Young (18-25 years)
  - Adult (26-35 years) 
  - Middle (36-50 years)
  - Senior (51+ years)
* Created `BMI_Category` column using standard classifications:
  - Underweight (BMI < 18.5)
  - Normal (18.5 ≤ BMI ≤ 24.9)
  - Overweight (25 ≤ BMI ≤ 29.9)
  - Obese (BMI ≥ 30)
* Verified new features using `value_counts()`

### 4. Exploratory Data Analysis (EDA)
* Conducted statistical analysis of numerical and categorical variables
* Analyzed average charges by region, smoker status, BMI category, and age group
* Examined correlations between age, BMI, children, and charges
* Created comprehensive visualizations:
  - Countplots for categorical distributions
  - Boxplots for charges across categories
  - Histogram of charges distribution
  - Correlation heatmap for numerical features
  - Pairplot with smoker status as hue

### 5. KPI Generation
* Calculated overall average medical cost
* Computed average charges by region, smoker status, BMI category, and age group
* Presented KPIs in structured markdown tables with insights

### 6. Machine Learning Model Development
* Applied one-hot encoding to categorical variables
* Defined features (X) and target variable (y = charges)
* Split data into training and testing sets (80/20)
* Trained two regression models:
  - Linear Regression
  - Random Forest Regressor
* Generated predictions on test data

### 7. Model Evaluation
* Evaluated models using key metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE) 
  - R² Score
* Created comparison table of model performance
* Analyzed reasons for performance differences

## Key Python Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

## Visualizations Overview

| Visualization Type | Description | Key Insights |
|-------------------|-------------|--------------|
| **Countplots** | Distribution of categorical variables (sex, smoker, region, BMI_Category) | Balanced gender distribution, majority non-smokers, Southeast has most entries |
| **Boxplots** | Charges vs Smoker, Region, BMI_Category | Smokers have significantly higher charges, Southeast region shows highest median costs |
| **Histogram** | Distribution of medical charges | Right-skewed distribution with most charges concentrated at lower values |
| **Correlation Heatmap** | Relationships between numerical features | Age shows moderate correlation (0.30) with charges, BMI weak correlation (0.20) |
| **Pairplot** | Relationships between key variables with smoker hue | Clear separation between smoker and non-smoker charges across all variables |

## Key Performance Indicators (KPIs)

### Overall Average Medical Charges
* **$13,279.12** - Average medical cost per individual

### Average Charges by Region
| Region | Average Charges |
|--------|----------------|
| Southeast | $14,735.41 |
| Northeast | $13,406.38 |
| Northwest | $12,450.84 |
| Southwest | $12,346.94 |

**Insight:** Southeast region has the highest average medical charges, while Southwest has the lowest, indicating regional variations in healthcare costs.

### Average Charges by Smoker Status
| Smoker | Average Charges |
|--------|----------------|
| Yes | $32,050.23 |
| No | $8,440.66 |

**Insight:** Smokers incur nearly **4x higher** medical charges than non-smokers, highlighting smoking as a major cost driver.

### Average Charges by BMI Category
| BMI Category | Average Charges |
|-------------|----------------|
| Obese | $15,510.92 |
| Overweight | $11,006.81 |
| Normal | $10,404.90 |
| Underweight | $8,657.62 |

**Insight:** Obese individuals face the highest medical costs, showing clear trend of increasing charges with higher BMI categories.

### Average Charges by Age Group
| Age Group | Average Charges |
|-----------|----------------|
| Senior | $18,084.99 |
| Middle | $14,029.99 |
| Adult | $10,495.16 |
| Young | $9,111.43 |

**Insight:** Medical charges consistently increase with age, with seniors facing costs nearly double those of young individuals.

## Model Performance Comparison

| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| Linear Regression | $4,177.05 | $5,956.34 | 0.81 |
| Random Forest Regressor | $2,637.99 | $4,702.45 | 0.88 |

**Insight:** Random Forest Regressor significantly outperforms Linear Regression across all metrics, demonstrating better predictive accuracy for medical charges.

## Major Insights

### [1] Dominant Cost Drivers
* **Smoking Status**: Most significant factor - smokers pay 4x more than non-smokers
* **Age**: Strong positive correlation (0.30) - costs increase steadily with age
* **BMI**: Clear trend of higher costs with increasing BMI categories
* **Region**: Southeast has highest average costs, Southwest the lowest

### [2] Data Distribution Patterns
* **Charges Distribution**: Highly right-skewed with most individuals at lower cost range
* **BMI Categories**: Majority of population in Obese (715) and Overweight (380) categories
* **Age Groups**: Relatively balanced distribution across age cohorts

### [3] Correlation Insights
* Age and Charges: Moderate correlation (0.30)
* BMI and Charges: Weak correlation (0.20) 
* Children and Charges: Very weak correlation (0.07)

### [4] Model Performance
* Random Forest handles non-linear relationships better than Linear Regression
* 88% variance explained by Random Forest vs 81% by Linear Regression
* Significant reduction in prediction errors with ensemble method

## Real-World Implications

### For Insurance Companies
* **Risk Assessment**: Use smoking status, age, and BMI for accurate premium calculation
* **Targeted Products**: Develop specialized insurance products for high-risk groups
* **Preventive Programs**: Invest in wellness programs to reduce long-term costs

### For Healthcare Providers
* **Resource Planning**: Allocate resources based on demographic risk factors
* **Preventive Care**: Focus on smoking cessation and weight management programs
* **Regional Strategy**: Address geographical variations in healthcare costs

### For Public Health Policy
* **Smoking Cessation**: Prioritize anti-smoking campaigns given massive cost impact
* **Obesity Prevention**: Implement public health initiatives for weight management
* **Age-Specific Care**: Develop targeted healthcare strategies for different age groups

## Technical Implementation

The project demonstrates professional data science workflow including:
* Comprehensive data cleaning and validation
* Strategic feature engineering
* Multi-faceted exploratory analysis
* Comparative model development and evaluation
* Business-focused insights generation

---

**Tools Used**: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn  
**Dataset**: Medical Cost Personal Dataset from Kaggle  
**Analysis Type**: Regression, Predictive Modeling, Business Intelligence
