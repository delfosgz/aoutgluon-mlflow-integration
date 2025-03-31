import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fix_day_sequence_with_month_year(df, day_column='day', cycle_length=365):
    """
    Fixes day sequence anomalies and adds month and year columns.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with day column
    day_column (str): Name of column containing day values (default: 'day')
    cycle_length (int): Length of the cycle (default: 365 for year)
    
    Returns:
    pandas.DataFrame: DataFrame with corrected day sequence, month, year, and datetime columns
    """
    # Create a copy of the DataFrame
    df_fixed = df.copy()
    days = df_fixed[day_column].values
    
    # Initialize arrays for corrected days, months, and years
    fixed_days = np.zeros_like(days)
    months = np.zeros_like(days)
    years = np.ones_like(days)  # Start with year 1
    
    # Days in each month (non-leap year)
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # Set first day
    current_day = 1
    current_month = 1
    current_year = 1
    day_of_year = 1
    
    fixed_days[0] = current_day
    months[0] = current_month
    years[0] = current_year
    
    # Process each day
    for i in range(1, len(days)):
        # Calculate expected next day
        expected_next = current_day + 1 if current_day < cycle_length else 1
        
        # Current value from data
        current_value = days[i]
        
        # Check if current value follows the sequence
        if (current_value == expected_next) or \
        (abs(current_value - expected_next) <= 5 and current_value <= cycle_length) or \
        (current_day == cycle_length and current_value == 1):
            current_day = current_value
        else:
            current_day = expected_next
            
        day_of_year = current_day if current_day != 1 else 1
        if current_day == 1 and i > 0:
            current_year += 1
        
        # Calculate month based on day of year
        cumulative_days = 0
        for month, days_in_month in enumerate(days_per_month, 1):
            if day_of_year <= cumulative_days + days_in_month:
                current_month = month
                break
            cumulative_days += days_in_month
        
        fixed_days[i] = current_day
        months[i] = current_month
        years[i] = current_year
    
    # Assign corrected values to DataFrame
    df_fixed[day_column] = fixed_days
    df_fixed['month'] = months
    df_fixed['year'] = years
    # Add datetime column
    #df_fixed['datetime'] = pd.to_datetime(df_fixed[['year', 'month', day_column]])
    
    return df_fixed

def feature_engineering(df):
    """
    Create new features based on meteorological understanding and data analysis,
    with 'day' representing day of the year (1-365).
    Ensures no data leakage by avoiding use of the target variable (rainfall).
    """
    # Make a copy to avoid modifying the original dataframe
    enhanced_df = df.copy()
    
    # 1. temparature range (difference between max and min temparatures)
    enhanced_df['temp_range'] = enhanced_df['maxtemp'] - enhanced_df['mintemp']
    
    # 2. Dew point depression (difference between temparature and dew point)
    enhanced_df['dewpoint_depression'] = enhanced_df['temparature'] - enhanced_df['dewpoint']
    
    # 3. Pressure change from previous day
    enhanced_df['pressure_change'] = enhanced_df['pressure'].diff().fillna(0)
    
    # 4. Humidity to dew point ratio
    enhanced_df['humidity_dewpoint_ratio'] = enhanced_df['humidity'] / enhanced_df['dewpoint'].clip(lower=0.1)
    
    # 5. Cloud coverage to sunshine ratio (inverse relationship)
    enhanced_df['cloud_sunshine_ratio'] = enhanced_df['cloud'] / enhanced_df['sunshine'].clip(lower=0.1)
    
    # 6. Wind intensity factor (combination of speed and humidity)
    enhanced_df['wind_humidity_factor'] = enhanced_df['windspeed'] * (enhanced_df['humidity'] / 100)
    
    # 7. temparature-humidity index (simple version of heat index)
    enhanced_df['temp_humidity_index'] = (0.8 * enhanced_df['temparature']) + \
                                        ((enhanced_df['humidity'] / 100) * \
                                        (enhanced_df['temparature'] - 14.3)) + 46.4
    
    # 8. Pressure change rate (acceleration)
    enhanced_df['pressure_acceleration'] = enhanced_df['pressure_change'].diff().fillna(0)
    
    # 9. Seasonal features (based on day of year)
    # Convert day to month (1-365 to 1-12)
    enhanced_df['month'] = ((enhanced_df['day'] - 1) // 30) + 1
    enhanced_df['month'] = enhanced_df['month'].clip(upper=12)  # Ensure month doesn't exceed 12
    
    # 10. Convert day to season (1-365 to 1-4)
    enhanced_df['season'] = ((enhanced_df['month'] - 1) // 3) + 1
    
    # 11. Sine and cosine transformations to capture cyclical nature of days in a year
    enhanced_df['day_of_year_sin'] = np.sin(2 * np.pi * enhanced_df['day'] / 365)
    enhanced_df['day_of_year_cos'] = np.cos(2 * np.pi * enhanced_df['day'] / 365)
    
    # 12. Rolling averages for key meteorological variables
    for window in [3, 7, 14]:
        enhanced_df[f'temparature_rolling_{window}d'] = enhanced_df['temparature'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'pressure_rolling_{window}d'] = enhanced_df['pressure'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'humidity_rolling_{window}d'] = enhanced_df['humidity'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'cloud_rolling_{window}d'] = enhanced_df['cloud'].rolling(window=window, min_periods=1).mean()
        enhanced_df[f'windspeed_rolling_{window}d'] = enhanced_df['windspeed'].rolling(window=window, min_periods=1).mean()
    
    # 13. Weather pattern change features
    # temparature trend
    enhanced_df['temp_trend_3d'] = enhanced_df['temparature'].diff(3).fillna(0)
    # Pressure trend
    enhanced_df['pressure_trend_3d'] = enhanced_df['pressure'].diff(3).fillna(0)
    # Humidity trend
    enhanced_df['humidity_trend_3d'] = enhanced_df['humidity'].diff(3).fillna(0)
    
    # 14. Extreme weather indicators
    enhanced_df['extreme_temp'] = (enhanced_df['temparature'] > enhanced_df['temparature'].quantile(0.95)) | \
                                 (enhanced_df['temparature'] < enhanced_df['temparature'].quantile(0.05))
    enhanced_df['extreme_temp'] = enhanced_df['extreme_temp'].astype(int)
    
    enhanced_df['extreme_humidity'] = (enhanced_df['humidity'] > enhanced_df['humidity'].quantile(0.95)) | \
                                     (enhanced_df['humidity'] < enhanced_df['humidity'].quantile(0.05))
    enhanced_df['extreme_humidity'] = enhanced_df['extreme_humidity'].astype(int)
    
    enhanced_df['extreme_pressure'] = (enhanced_df['pressure'] > enhanced_df['pressure'].quantile(0.95)) | \
                                     (enhanced_df['pressure'] < enhanced_df['pressure'].quantile(0.05))
    enhanced_df['extreme_pressure'] = enhanced_df['extreme_pressure'].astype(int)
    
    # 15. Interaction terms between key variables
    enhanced_df['temp_humidity_interaction'] = enhanced_df['temparature'] * enhanced_df['humidity']
    enhanced_df['pressure_wind_interaction'] = enhanced_df['pressure'] * enhanced_df['windspeed']
    enhanced_df['cloud_sunshine_interaction'] = enhanced_df['cloud'] * enhanced_df['sunshine']
    enhanced_df['dewpoint_humidity_interaction'] = enhanced_df['dewpoint'] * enhanced_df['humidity']
    
    # 16. Moving standard deviations for measuring variability
    for window in [7, 14]:
        enhanced_df[f'temp_std_{window}d'] = enhanced_df['temparature'].rolling(window=window, min_periods=4).std().fillna(0)
        enhanced_df[f'pressure_std_{window}d'] = enhanced_df['pressure'].rolling(window=window, min_periods=4).std().fillna(0)
        enhanced_df[f'humidity_std_{window}d'] = enhanced_df['humidity'].rolling(window=window, min_periods=4).std().fillna(0)
    
    return enhanced_df

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def show_violin_plot(data, columns_to_plot, target):
    """
    Creates target separate violin plots for each feature in columns_to_plot.
    The left column displays a violin plot, excluding the target from columns_to_plot.
    
    Parameters:
    data (pandas.DataFrame): The DataFrame containing the data.
    columns_to_plot (list): List of column names to visualize (excluding target).
    target (str): The name of the target variable for grouping.
    """
    # Ensure the target is excluded from the columns_to_plot list
    columns_to_plot = [col for col in columns_to_plot if col != target]

    # Define a more visually appealing color palette
    custom_palette = ["#3498db", "#e74c3c"]  # Professional blue and red

    # Set figure style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

    # Calculate number of rows and columns for the subplot grid
    n_columns = 3  # Three plots per row
    n_rows = int(np.ceil(len(columns_to_plot) / n_columns))  # Calculate rows needed

    # Create figure with the appropriate number of rows and columns
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(5 * n_columns, 3 * n_rows), dpi=100)

    # Flatten axs to handle the grid easily
    axs = axs.flatten()

    # Set overall figure background
    fig.patch.set_facecolor('#f8f9fa')

    # Loop through each feature and create the violin plot
    for i, col in enumerate(columns_to_plot):
        # Left: Violin plot
        sns.violinplot(
            x=target, 
            y=col, 
            data=data, 
            ax=axs[i],
            hue=target,  # Assign target to hue
            palette=custom_palette,
            inner='quartile',
            linewidth=1.5,
            cut=0,
            legend=False  # Turn off the legend
        )
        axs[i].set_title(f'{col.title()} Distribution', 
                         fontsize=8, fontweight='bold', pad=5)  # Reduced padding
        axs[i].set_xlabel('')
        axs[i].set_ylabel(col.title(), fontsize=12, labelpad=10)
        axs[i].tick_params(labelsize=11)
        axs[i].grid(axis='y', linestyle='--', alpha=0.7)

        # Set x-tick positions and labels explicitly
        axs[i].set_xticks([0, 1])  # Set the positions explicitly
        axs[i].set_xticklabels(['No Rain (0)', 'Rain (1)'])

        # Remove unnecessary spines for a cleaner look
        sns.despine(ax=axs[i])
    
    # Turn off any remaining unused axes
    for i in range(len(columns_to_plot), len(axs)):
        axs[i].axis('off')

    plt.subplots_adjust(hspace=0.5)
    plt.show()
    return None
