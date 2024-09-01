import pandas as pd



# Load the datasets
monthly_CMO_data = pd.read_csv('CMO-Historical-Data-Monthly.csv', skiprows=4, encoding='UTF-8')
exchange_rate_data = pd.read_csv('EXCHUS_FRED.csv', skiprows=0, encoding='UTF-8')
monthly_NBS_data = pd.read_csv('Monthly_NBS.csv', skiprows=2, encoding='UTF-8')
quarterly_NBS_data = pd.read_csv('Quarterly_NBS.csv', skiprows=2, encoding='UTF-8')
world_bank_df = pd.read_csv('World_Development_Indicators.csv', skiprows=0, encoding='UTF-8')



# Iron ore Data Preprocessing (Monthly data)
iron_ore_df = monthly_CMO_data.loc[5:, ['Unnamed: 0', 'Iron ore, cfr spot', 'Crude oil, Brent', 'Coal, Australian']]
iron_ore_df.rename(columns={'Unnamed: 0': 'Date', 'Iron ore, cfr spot' :'Iron ore, cfr spot ($/dmtu)',
                            'Crude oil, Brent':'Crude oil, Brent ($/bbl)','Coal, Australian':'Coal, Australian ($/mt)'}, inplace=True)
iron_ore_df['Date'] = pd.to_datetime(iron_ore_df['Date'], format='%YM%m')
iron_ore_df.set_index('Date', inplace=True)
print(iron_ore_df)



# FRED Exchange Data Preprocessing (Monthly data)
exchange_rate_data.rename(columns={'DATE': 'Date'}, inplace=True)
exchange_rate_data['Date'] = pd.to_datetime(exchange_rate_data['Date'], format='%d/%m/%Y')
exchange_rate_data.set_index('Date', inplace=True)



# Monthly NBS Data Preprocessing (Monthly data)
monthly_NBS_data = monthly_NBS_data.drop([7,8,9,10])
monthly_NBS_data = monthly_NBS_data.transpose()
monthly_NBS_data.columns = monthly_NBS_data.iloc[0]
monthly_NBS_data = monthly_NBS_data.drop(monthly_NBS_data.index[0])
monthly_NBS_data.reset_index(inplace=True)
monthly_NBS_data.rename(columns={'index': 'Date'}, inplace=True)
monthly_NBS_data['Date'] = pd.to_datetime(monthly_NBS_data['Date'], format='%b-%y')
monthly_NBS_data.set_index('Date', inplace=True)



# Quarterly NBS Data Preprocessing (Monthly data)
quarterly_NBS_data = quarterly_NBS_data.drop([3, 4, 5, 6, 7])
quarterly_NBS_data = quarterly_NBS_data.transpose()
quarterly_NBS_data.columns = quarterly_NBS_data.iloc[0]
quarterly_NBS_data = quarterly_NBS_data.drop(quarterly_NBS_data.index[0])
quarterly_NBS_data.reset_index(inplace=True)
quarterly_NBS_data.rename(columns={'index': 'Date'}, inplace=True)



# Function to convert quarter to date
def convert_quarter_to_date(quarter_str):
    try:
        q, year = quarter_str.split(' ')
        quarter = int(q[0])
        month = (quarter - 1) * 3 + 1
        return pd.Timestamp(f'{year}-{month:02d}-01')
    except Exception as e:
        print(f"Error converting {quarter_str}: {e}")
        return pd.NaT

quarterly_NBS_data['Date'] = quarterly_NBS_data['Date'].apply(convert_quarter_to_date)
quarterly_NBS_data = quarterly_NBS_data.dropna(subset=['Date'])  # Drop rows with invalid dates
quarterly_NBS_data.set_index('Date', inplace=True)



# WB Indicators Filtering
series_to_extract = [
    'Foreign direct investment, net inflows (% of GDP)',
    'Industry (including construction), value added (% of GDP)',
    'CO2 emissions (metric tons per capita)',
    'Inflation, GDP deflator (annual %)', 
    'Inflation, consumer prices (annual %)',
    'Trade (% of GDP)', 
    'CO2 emissions from manufacturing industries and construction (% of total fuel combustion)',
    'Tariff rate, applied, simple mean, primary products (%)',
    'GNI growth (annual %)'
]

china_data = world_bank_df[world_bank_df['Series Name'].isin(series_to_extract)]
china_data.set_index(['Series Name', 'Country Name'], inplace=True)
china_data = china_data.transpose()
china_data = china_data.drop(['Country Code', 'Series Code'], axis=0)
china_data.columns = china_data.columns.droplevel('Country Name')
china_data.index.name = 'Date'
china_data.index = pd.to_datetime(china_data.index.str.split(' ').str[0])



# Combine all the DFs
merged_df = pd.concat([iron_ore_df, exchange_rate_data, monthly_NBS_data, quarterly_NBS_data, china_data], axis=1, join='outer')

merged_df.drop(columns=[
    'Number of Projects for Contracted Foreign Direct Investment Accumulated Growth Rate(%)',
    'Total Income of Construction Enterprises Accumulated(100 million yuan)',
    'CO2 emissions (metric tons per capita)',
    'CO2 emissions from manufacturing industries and construction (% of total fuel combustion)',
    'Tariff rate, applied, simple mean, primary products (%)'
], inplace=True)

# Create a date range from 01/01/2008 to 01/04/2024
date_range = pd.date_range(start='2007-12-01', end='2024-04-01', freq='MS')

# Reindex the merged DataFrame with this date range
merged_df = merged_df.reindex(date_range)

# Forward fill the values
merged_df.ffill(inplace=True)
merged_df.index.name = 'Date'

print(merged_df)

merged_df.to_csv('Monthly_Price_Determinant_Data_v3.csv')
