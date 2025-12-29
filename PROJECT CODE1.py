import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ============ LOAD DATA ====================
Orders = pd.read_excel("C:/Users/User/Desktop/327 Project/05_Project_DT_C/05_Project_DT_C.xlsx", sheet_name="Orders")
DistanceMatrix = pd.read_excel("C:/Users/User/Desktop/327 Project/05_Project_DT_C/05_Project_DT_C.xlsx", sheet_name="DistanceMatrix")
DeliveryLogs = pd.read_excel("C:/Users/User/Desktop/327 Project/05_Project_DT_C/05_Project_DT_C.xlsx", sheet_name="DeliveryLogs")

# ============ PREPARE DISTANCE MATRIX =======
distances = []
for i in range(len(DistanceMatrix)):
    from_city = DistanceMatrix.iloc[i, 0]
    for to_city in DistanceMatrix.columns[1:]:
        distance = DistanceMatrix.at[i, to_city]
        if pd.notna(distance):
            distances.append({'From': from_city, 'To': to_city, 'Distance': distance})
distances_df = pd.DataFrame(distances)

# ============ PROCESS DELIVERY LOGS =========
DeliveryLogs['Time'] = pd.to_datetime(DeliveryLogs['Time'])
lead_time_data = DeliveryLogs.pivot_table(
    index='ID', columns='Status', values='Time', aggfunc='first'
).reset_index()
lead_time_data['LeadTime'] = (lead_time_data['Delivered'] - lead_time_data['Ordered']).dt.total_seconds() / (24 * 3600)  # Convert to days as float

# ============ MERGE DATA ====================
OrdersWithLeadTime = Orders.merge(lead_time_data[['ID','LeadTime']], on='ID', how='left')
merged_df = OrdersWithLeadTime.merge(distances_df, on=['From','To'], how='left')

# ============ SPLIT DATA 70/30 ============
size70 = round(len(merged_df) * 0.7)
df70 = merged_df.iloc[:size70].copy()
df30 = merged_df.iloc[size70:].copy()  

# ============ CLEAN DATA ===================
# Handle missing values more appropriately
df70['LeadTime'] = df70['LeadTime'].fillna(df70['LeadTime'].median())  # Use median instead of 0
df70['OrderVolume'] = df70['OrderVolume'].fillna(0)
df70['Distance'] = df70['Distance'].fillna(df70['Distance'].median())  # Use median instead of 0

df30['LeadTime'] = df30['LeadTime'].fillna(df70['LeadTime'].median())  # Use training median
df30['OrderVolume'] = df30['OrderVolume'].fillna(0)
df30['Distance'] = df30['Distance'].fillna(df70['Distance'].median())  # Use training median

# ============ LINEAR REGRESSION ========
target = "LeadTime"
numeric_cols = df70.select_dtypes(include=['float64','int64']).columns.drop(target)
formula = f"{target} ~ {' + '.join(numeric_cols)} + C(DestinationType)"
Lm = smf.ols(formula=formula, data=df70).fit()
print(Lm.summary())

# Prediction on 30% test data
Lm_Predict = Lm.predict(df30)

# ============ RANDOM FOREST ===========
df_train = df70.copy()
df_test = df30.copy()

for df in [df_train, df_test]:
    if 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

label_encoders = {}
for col in ['From','To','DestinationType']:
    le = LabelEncoder()
    combined = pd.concat([df_train[col], df_test[col]], axis=0)
    le.fit(combined)
    df_train[col] = le.transform(df_train[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le

X_train = df_train.drop(columns=['LeadTime'])
y_train = df_train['LeadTime']
X_test = df_test.drop(columns=['LeadTime'])
y_test = df_test['LeadTime']

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict
y_pred_rf = rf.predict(X_test)

# ============ MODEL COMPARISON ============
mse_lr = mean_squared_error(y_test, Lm_Predict)
r2_lr = r2_score(y_test, Lm_Predict)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    if sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape_lr = calculate_mape(y_test.values, Lm_Predict.values)
mape_rf = calculate_mape(y_test.values, y_pred_rf)

print("========== MODEL COMPARISON ==========")
print(f"Linear Regression - R²: {r2_lr:.4f}, MSE: {mse_lr:.4f}, MAPE: {mape_lr:.2f}%")
print(f"Random Forest     - R²: {r2_rf:.4f}, MSE: {mse_rf:.4f}, MAPE: {mape_rf:.2f}%")


# ============PREDICT & COMPARE 2 OBSERVATIONS FROM DF30 BY ID ============

print("Available columns in df30:", df30.columns.tolist())

df_test_with_id = df30.copy()  

X_test_with_id = df_test_with_id.drop(columns=['LeadTime', 'ID'], errors='ignore')

for col in ['From', 'To', 'DestinationType']:
    if col in X_test_with_id.columns:
        X_test_with_id[col] = label_encoders[col].transform(X_test_with_id[col])

y_test_with_id = df_test_with_id['LeadTime']

sample_ids = df_test_with_id['ID'].head(2).values
print(f"\nSelected IDs for prediction: {sample_ids}")

comparison_results = []

for i, sample_id in enumerate(sample_ids):
    
    sample_row = df_test_with_id[df_test_with_id['ID'] == sample_id].iloc[0]
    
    features_for_pred = sample_row.drop(['ID', 'LeadTime'])
    
    encoded_features = features_for_pred.copy()
    for col in ['From', 'To', 'DestinationType']:
        if col in encoded_features.index:
            encoded_features[col] = label_encoders[col].transform([encoded_features[col]])[0]
    
    features_df = pd.DataFrame([encoded_features])
    
    features_df = features_df[X_train.columns]
    
    predicted_lead_time = rf.predict(features_df)[0]
    actual_lead_time = sample_row['LeadTime']
    
    error = predicted_lead_time - actual_lead_time
    abs_error = abs(error)
    pct_error = (abs_error / actual_lead_time * 100) if actual_lead_time != 0 else np.nan
    
    result = {
        'ID': sample_id,
        'From': sample_row['From'],
        'To': sample_row['To'],
        'DestinationType': sample_row['DestinationType'],
        'OrderVolume': sample_row['OrderVolume'],
        'Distance': sample_row['Distance'],
        'Actual_LeadTime': actual_lead_time,
        'Predicted_LeadTime': predicted_lead_time,  # This is the key addition
        'Difference': error,
        'Absolute_Error': abs_error,
        'Percentage_Error': pct_error
    }
    comparison_results.append(result)
    
    print(f"\n" + "="*60)
    print(f"OBSERVATION {i+1} - ID: {sample_id}")
    print("="*60)
    print(f"Actual Lead Time: {actual_lead_time:.4f} days")
    print(f"Predicted Lead Time: {predicted_lead_time:.4f} days")  # Explicitly shown
    print(f"Error: {error:.4f} days")
    print(f"Absolute Error: {abs_error:.4f} days")
    if actual_lead_time != 0:
        print(f"Percentage Error: {pct_error:.2f}%")
    else:
        print(f"Percentage Error: N/A (actual value is 0)")
    
    print(f"\nFeature Values:")
    print(f"  From: {sample_row['From']}")
    print(f"  To: {sample_row['To']}")
    print(f"  DestinationType: {sample_row['DestinationType']}")
    print(f"  OrderVolume: {sample_row['OrderVolume']}")
    print(f"  Distance: {sample_row['Distance']:.2f}")

# Create a comparison DataFrame
comparison_df = pd.DataFrame(comparison_results)

column_order = ['ID', 'From', 'To', 'DestinationType', 'OrderVolume', 'Distance', 
                'Actual_LeadTime', 'Predicted_LeadTime', 'Difference', 'Absolute_Error', 'Percentage_Error']
comparison_df = comparison_df[column_order]

print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON DATAFRAME WITH PREDICTED LEAD TIMES")
print("="*80)
print(comparison_df.round(4))

print("\n" + "="*80)
print("KEY PREDICTION COMPARISON")
print("="*80)
key_columns = ['ID', 'From', 'To', 'Actual_LeadTime', 'Predicted_LeadTime', 'Difference', 'Absolute_Error']
print(comparison_df[key_columns].round(4))

mse_samples = mean_squared_error(comparison_df['Actual_LeadTime'], comparison_df['Predicted_LeadTime'])
rmse_samples = np.sqrt(mse_samples)
mae_samples = np.mean(comparison_df['Absolute_Error'])

print("\n" + "="*80)
print("PERFORMANCE METRICS FOR THESE 2 OBSERVATIONS")
print("="*80)
print(f"Mean Squared Error (MSE): {mse_samples:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_samples:.4f}")
print(f"Mean Absolute Error (MAE): {mae_samples:.4f}")

plt.figure(figsize=(15, 6))

# Plot: Actual vs Predicted comparison
plt.subplot(1, 3, 1)
x_pos = np.arange(len(comparison_df))
width = 0.35

bars_actual = plt.bar(x_pos - width/2, comparison_df['Actual_LeadTime'], width, 
                     label='Actual', alpha=0.7, color='blue', edgecolor='black')
bars_pred = plt.bar(x_pos + width/2, comparison_df['Predicted_LeadTime'], width, 
                   label='Predicted', alpha=0.7, color='red', edgecolor='black')

plt.xlabel('Observation ID')
plt.ylabel('Lead Time (days)')
plt.title('Actual vs Predicted Lead Times\n(Random Forest Model)')
plt.xticks(x_pos, [f'ID: {id}' for id in comparison_df['ID']])
plt.legend()
plt.grid(True, alpha=0.3)


print("\n" + "="*80)
print("PREDICTION PERFORMANCE ANALYSIS")
print("="*80)
for i, row in comparison_df.iterrows():
    accuracy = 100 - abs(row['Percentage_Error']) if not np.isnan(row['Percentage_Error']) else 'N/A'
    status = "GOOD" if row['Absolute_Error'] < 1 else "MODERATE" if row['Absolute_Error'] < 2 else "POOR"
    
    print(f"\nID {row['ID']}:")  
    print(f"  Actual: {row['Actual_LeadTime']:.2f} days, Predicted: {row['Predicted_LeadTime']:.2f} days")
    print(f"  Error: {row['Difference']:.2f} days | Absolute Error: {row['Absolute_Error']:.2f} days")
    print(f"  Accuracy: {accuracy}% | Status: {status}")


# ============CORRELATION ANALYSIS ============

print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

correlation_matrix = analysis_df[numerical_cols].corr()

print("Correlation Matrix:")
print(correlation_matrix.round(3))

plt.figure(figsize=(12, 10))

mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True,
            mask=mask,
            fmt='.3f',
            cbar_kws={'shrink': 0.8})

plt.title('Correlation Matrix Heatmap\n(Numerical Variables)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("CORRELATIONS WITH LEAD TIME (Target Variable)")
print("="*80)

if 'LeadTime' in correlation_matrix.columns:
    lead_time_correlations = correlation_matrix['LeadTime'].sort_values(ascending=False)
    print(lead_time_correlations.round(3))

print("\n" + "="*80)
print("SCATTER PLOT MATRIX - KEY VARIABLES")
print("="*80)

key_vars_for_pairplot = ['LeadTime', 'Distance', 'OrderVolume']
if all(var in analysis_df.columns for var in key_vars_for_pairplot):
    pairplot_data = analysis_df[key_vars_for_pairplot].dropna()
    
    pair_grid = sns.PairGrid(pairplot_data)
    pair_grid.map_upper(plt.scatter, alpha=0.6)
    pair_grid.map_lower(sns.kdeplot, fill=True)
    pair_grid.map_diag(plt.hist, alpha=0.7)
    
    plt.suptitle('Pair Plot of Key Variables', y=1.02, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============ EXPLANATION OF CORRELATION ANALYSIS ============

print("\n" + "="*80)
print("EXPLANATION OF CORRELATION ANALYSIS")
print("="*80)



# ============ ADDITIONAL CORRELATION VISUALIZATION ============

plt.figure(figsize=(10, 6))

if 'LeadTime' in correlation_matrix.columns:
    # Get correlations with LeadTime, excluding itself
    lead_time_corr = correlation_matrix['LeadTime'].drop('LeadTime', errors='ignore')
    lead_time_corr_sorted = lead_time_corr.sort_values(ascending=True)
    
    colors = ['red' if x < 0 else 'blue' for x in lead_time_corr_sorted]
    plt.barh(range(len(lead_time_corr_sorted)), lead_time_corr_sorted, color=colors, alpha=0.7)
    plt.yticks(range(len(lead_time_corr_sorted)), lead_time_corr_sorted.index)
    plt.xlabel('Correlation Coefficient with Lead Time')
    plt.title('Variable Correlations with Lead Time (Target Variable)', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(lead_time_corr_sorted):
        plt.text(v, i, f' {v:.3f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

print("\n" + "="*80)
print("SUMMARY OF KEY FINDINGS")
print("="*80)



# ============CORRELATION(LINEAR REGRESSION MODEL) WITH TARGET VARIABLE ============

print("\n2. CORRELATION WITH TARGET VARIABLE (LeadTime)")
print("-" * 50)

if 'LeadTime' in correlation_matrix.columns:
    lead_time_correlations = correlation_matrix['LeadTime'].drop('LeadTime').sort_values(ascending=False)
    print("Correlation with LeadTime:")
    for var, corr in lead_time_correlations.items():
        strength = "STRONG" if abs(corr) > 0.7 else "MODERATE" if abs(corr) > 0.5 else "WEAK" if abs(corr) > 0.3 else "VERY WEAK"
        direction = "positive" if corr > 0 else "negative"
        print(f"  {var}: {corr:.3f} ({strength} {direction} correlation)")
        
# ============SCATTER MATRIX FOR LINEAR REGRESSION MODEL ============

print("="*80)
print("SCATTER MATRIX FOR LINEAR REGRESSION MODEL")
print("="*80)

scatter_vars = ['LeadTime', 'OrderVolume', 'Distance']
scatter_df = lr_analysis_df[scatter_vars].dropna()

print(f"Variables included in scatter matrix: {scatter_vars}")
print(f"Number of observations: {len(scatter_df)}")

# ============ BASIC SCATTER MATRIX ============

print("\n1. BASIC SCATTER MATRIX PLOT")
print("-" * 40)

plt.figure(figsize=(15, 12))
pd.plotting.scatter_matrix(scatter_df, alpha=0.7, figsize=(15, 12), diagonal='hist',
                          hist_kwds={'bins': 20, 'alpha': 0.8, 'edgecolor': 'black'},
                          s=50, color='steelblue')
plt.suptitle('Scatter Matrix for Linear Regression Variables\n(Diagonal: Histograms, Off-diagonal: Scatter plots)', 
             fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
plt.show()

# ============ ENHANCED SCATTER MATRIX WITH SEABORN ============

print("\n2. ENHANCED SCATTER MATRIX WITH CORRELATION COEFFICIENTS")
print("-" * 55)

def corrfunc(x, y, **kws):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.annotate(f'r = {r:.2f}', xy=(0.1, 0.9), xycoords=ax.transAxes,
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.figure(figsize=(15, 12))
g = sns.PairGrid(scatter_df, diag_sharey=False)
g.map_upper(sns.scatterplot, alpha=0.7, color='steelblue', s=60)
g.map_upper(corrfunc)  # Add correlation coefficients to upper triangle
g.map_lower(sns.regplot, scatter_kws={'alpha': 0.6, 's': 50, 'color': 'steelblue'},
           line_kws={'color': 'red', 'linewidth': 2, 'alpha': 0.8})
g.map_diag(sns.histplot, kde=True, color='steelblue', alpha=0.8, edgecolor='black')

for i, ax in enumerate(g.axes.flat):
    if i % len(scatter_vars) == 0:  # First column
        ax.set_ylabel(scatter_vars[i // len(scatter_vars)], fontweight='bold')
    if i >= len(scatter_vars) * (len(scatter_vars) - 1):  # Last row
        ax.set_xlabel(scatter_vars[i % len(scatter_vars)], fontweight='bold')

plt.suptitle('Enhanced Scatter Matrix with Regression Lines and Correlation Coefficients', 
             fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
plt.show()

# ============ SCATTER MATRIX WITH DESTINATION TYPE COLOR CODING ============

print("\n3. SCATTER MATRIX COLORED BY DESTINATION TYPE")
print("-" * 50)

if 'DestinationType' in lr_analysis_df.columns:
    # Add DestinationType to scatter data
    scatter_df_with_type = scatter_df.copy()
    scatter_df_with_type['DestinationType'] = lr_analysis_df['DestinationType']
    
    plt.figure(figsize=(15, 12))
    g = sns.PairGrid(scatter_df_with_type, hue='DestinationType', diag_sharey=False)
    g.map_upper(sns.scatterplot, alpha=0.7, s=50)
    g.map_lower(sns.scatterplot, alpha=0.7, s=50)
    g.map_diag(sns.histplot, alpha=0.7, edgecolor='black', multiple="layer")
    g.add_legend(title='Destination Type')
    
    for i, ax in enumerate(g.axes.flat):
        if i % len(scatter_vars) == 0:
            ax.set_ylabel(scatter_vars[i // len(scatter_vars)], fontweight='bold')
        if i >= len(scatter_vars) * (len(scatter_vars) - 1):
            ax.set_xlabel(scatter_vars[i % len(scatter_vars)], fontweight='bold')
    
    plt.suptitle('Scatter Matrix Colored by Destination Type', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()

# ============ DENSITY-BASED SCATTER MATRIX ============

print("\n4. DENSITY-BASED SCATTER MATRIX")
print("-" * 35)

plt.figure(figsize=(15, 12))
g = sns.PairGrid(scatter_df)
g.map_upper(sns.scatterplot, alpha=0.6, s=40, color='blue')
g.map_lower(sns.kdeplot, fill=True, cmap='Blues', alpha=0.7)
g.map_diag(sns.histplot, kde=True, color='blue', alpha=0.7, edgecolor='black')

for i, ax in enumerate(g.axes.flat):
    if i % len(scatter_vars) == 0:
        ax.set_ylabel(scatter_vars[i // len(scatter_vars)], fontweight='bold')
    if i >= len(scatter_vars) * (len(scatter_vars) - 1):
        ax.set_xlabel(scatter_vars[i % len(scatter_vars)], fontweight='bold')

plt.suptitle('Density-Based Scatter Matrix\n(Upper: Scatter plots, Lower: Density contours)', 
             fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
plt.show()

# ============ SCATTER MATRIX WITH RESIDUAL ANALYSIS ============

print("\n5. SCATTER MATRIX WITH RESIDUAL ANALYSIS")
print("-" * 45)

if 'Residuals' not in lr_analysis_df.columns:
    lr_analysis_df['Predicted'] = Lm.predict(lr_analysis_df)
    lr_analysis_df['Residuals'] = lr_analysis_df['LeadTime'] - lr_analysis_df['Predicted']

residual_vars = ['Residuals', 'OrderVolume', 'Distance']
residual_scatter_df = lr_analysis_df[residual_vars].dropna()

plt.figure(figsize=(12, 10))
g = sns.PairGrid(residual_scatter_df, diag_sharey=False)
g.map_upper(sns.scatterplot, alpha=0.7, s=50, color='purple')
g.map_lower(sns.regplot, scatter_kws={'alpha': 0.6, 's': 40, 'color': 'purple'},
           line_kws={'color': 'red', 'linewidth': 2})
g.map_diag(sns.histplot, kde=True, color='purple', alpha=0.7, edgecolor='black')

def residual_corrfunc(x, y, **kws):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.annotate(f'r = {r:.3f}', xy=(0.1, 0.9), xycoords=ax.transAxes,
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

g.map_upper(residual_corrfunc)

plt.suptitle('Scatter Matrix: Residuals vs Predictors\n(Checking for Patterns in Residuals)', 
             fontsize=16, fontweight='bold', y=0.95)
plt.tight_layout()
plt.show()

# ============ INTERACTIVE-LIKE SCATTER MATRIX WITH MULTIPLE ELEMENTS ============

print("\n6. COMPREHENSIVE SCATTER MATRIX WITH MULTIPLE VISUALIZATION ELEMENTS")
print("-" * 70)

fig, axes = plt.subplots(len(scatter_vars), len(scatter_vars), figsize=(18, 15))

for i, row_var in enumerate(scatter_vars):
    for j, col_var in enumerate(scatter_vars):
        ax = axes[i, j]
        
        if i == j:
            sns.histplot(data=scatter_df, x=row_var, ax=ax, kde=True, 
                        color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_ylabel('Frequency')
            mean_val = scatter_df[row_var].mean()
            std_val = scatter_df[row_var].std()
            ax.text(0.05, 0.95, f'μ={mean_val:.1f}\nσ={std_val:.1f}', 
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
        else:
            sns.scatterplot(data=scatter_df, x=col_var, y=row_var, ax=ax, 
                           alpha=0.6, color='steelblue', s=50)
            
            z = np.polyfit(scatter_df[col_var], scatter_df[row_var], 1)
            p = np.poly1d(z)
            x_range = np.linspace(scatter_df[col_var].min(), scatter_df[col_var].max(), 100)
            ax.plot(x_range, p(x_range), 'r-', linewidth=2, alpha=0.8)
            
            corr = scatter_df[col_var].corr(scatter_df[row_var])
            ax.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            y_pred = p(scatter_df[col_var])
            ss_res = np.sum((scatter_df[row_var] - y_pred) ** 2)
            ss_tot = np.sum((scatter_df[row_var] - scatter_df[row_var].mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            ax.text(0.05, 0.85, f'R² = {r_squared:.2f}', transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        if i == len(scatter_vars) - 1:
            ax.set_xlabel(col_var, fontweight='bold')
        if j == 0:
            ax.set_ylabel(row_var, fontweight='bold')
        
        ax.grid(True, alpha=0.3)

plt.suptitle('Comprehensive Scatter Matrix for Linear Regression Analysis\n' +
            '(Includes Histograms, Scatter Plots, Regression Lines, Correlation, and R²)', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()