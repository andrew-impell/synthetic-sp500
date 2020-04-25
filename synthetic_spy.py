import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from tqdm import tqdm
import pmdarima
from pmdarima.arima import ndiffs
import pickle
from matplotlib import style

style.use('ggplot')

PERIODS = 50

cwd = os.getcwd()

market_cap = cwd + '/market.html'


def find_all(string, substring):
    """
    Function: Returning all the index of substring in a string
    Arguments: String and the search string
    Return:Returning a list
    """
    length = len(substring)
    c = 0
    indexes = []
    while c < len(string):
        if string[c:c+length] == substring:
            indexes.append(c)
        c = c+1
    return indexes


def get_dfs():

    spy_df = pd.read_csv(cwd + '/ETFs/spy.us.txt', header=0)
    spy_df['Date'] = pd.to_datetime(spy_df['Date'])
    spy_df.set_index('Date', inplace=True, drop=True)
    spy_df = spy_df['Close']

    ts_data_glob = glob(cwd + '/Stocks/*.txt')

    html_loc = cwd + '/sp500tickers.html'

    with open(html_loc, 'r') as myfile:
        ticker_data = myfile.read()

    dfs = pd.read_html(ticker_data)
    tickers = dfs[0]['Symbol'].values

    df_list = []
    total_tickers_read = 0

    for ticker in tqdm(tickers):
        for filename in ts_data_glob:
            first = filename.rfind('/')
            last = find_all(filename, '.')[-2]
            ticker_fname = str(filename[first+1:last].upper())

            if ticker_fname == ticker:
                df_out = pd.DataFrame()
                df_temp = pd.read_csv(filename, header=0)
                df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                df_temp.set_index('Date', inplace=True, drop=True)
                df_out[str(ticker) + ' Close'] = df_temp['Close']
                df_list.append(df_out)
                total_tickers_read += 1

    print(f'Tickers read: {total_tickers_read}')

    df = pd.concat(df_list, axis=1)

    print(f'Shape before cutting: {df.shape}')

    first_spy = spy_df.index.values[0]
    last_spy = spy_df.index.values[-1]

    df = df.loc[first_spy:last_spy, :]

    print(f'Shape after cutting: {df.shape}')

    df = df.dropna(axis='columns', how='any', thresh=3100)

    df.fillna(0)
    df.replace({np.inf: 0, -np.inf: 0})

    return df, spy_df


df, spy_df = get_dfs()


date_index = df.index

with open(market_cap, 'r') as myfile:
    ticker_data = myfile.read()

dfs = pd.read_html(ticker_data)
arr_ticks = dfs[0][['Company (Ticker)', 'Market Cap']]
arr_ticks['Ticker'] = arr_ticks['Company (Ticker)'].str.extract(r'\(([^)]+)')
arr_ticks.drop(['Company (Ticker)'], axis=1, inplace=True)
arr_ticks.drop(arr_ticks.tail(1).index, inplace=True)
arr_ticks.drop(arr_ticks.head(1).index, inplace=True)
arr_ticks.replace(regex=['\$'], value='', inplace=True)
arr_ticks.replace(regex=['billion'], value='', inplace=True)
arr_ticks = arr_ticks.astype({"Ticker": str, "Market Cap": np.float16})
market_sum = arr_ticks['Market Cap'].sum()
arr_ticks['Market pct'] = arr_ticks['Market Cap'] / market_sum
arr_ticks.drop(['Market Cap'], axis=1, inplace=True)
arr_ticks.set_index('Ticker', inplace=True)

sp_pct = arr_ticks.to_dict('index')


sp_balanced_arr = []


for col in tqdm(df.columns):
    prices = df.loc[:, col].values
    try:
        prices *= sp_pct[col[:-6]]['Market pct']
        price_df = pd.DataFrame(data=prices, index=date_index,
                                columns=[f'{col[:-6]}'])
        sp_balanced_arr.append(price_df)

    except KeyError:
        print(f'Could not process {col}')


sp_synthetic = pd.concat(sp_balanced_arr, axis=1)
sp_synthetic['Date'] = date_index
sp_synthetic.set_index('Date', inplace=True)
sp_synthetic['SP Synthetic'] = sp_synthetic.sum(axis=1)

spy_rescale = spy_df.mean() / sp_synthetic['SP Synthetic'].mean()


spy_df_reshape = np.append(spy_df.values, np.mean(spy_df.values[:-5]))

resid = (spy_rescale * sp_synthetic['SP Synthetic'].values) - spy_df_reshape

kpss_diffs = ndiffs(resid, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(resid, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print('='*35)
print('Fitting Residuals...')
print('='*35)


auto_res = pmdarima.auto_arima(resid, d=n_diffs, seasonal=False, stepwise=True,
                               suppress_warnings=True, error_action="ignore", max_p=6,
                               max_order=None, trace=True)

print(auto_res.order)
in_sample_resid = auto_res.predict_in_sample()


spy_synthetic_wresid = (spy_rescale * sp_synthetic['SP Synthetic'].values) - in_sample_resid
# plt.plot(sp_synthetic.index, spy_rescale * sp_synthetic['SP Synthetic'].values,
#          label='No Residuals SPY Synthetic', alpha=0.5)
plt.plot(sp_synthetic.index, spy_synthetic_wresid, label='SPY synthetic',
         alpha=0.5)
plt.plot(spy_df.index, spy_df.values, label='SPY', alpha=0.5)
plt.legend()
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('SPY vs Synthetically Constructed SPY')
plt.show()

predict_dict = {}
unforecast = []

for col in tqdm(df.columns):
    y_train = list(df.loc[:, col].values)
    try:
        kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
        n_diffs = max(adf_diffs, kpss_diffs)

        auto = pmdarima.auto_arima(y_train, d=n_diffs, seasonal=False, stepwise=True,
                                   suppress_warnings=True, error_action="ignore", max_p=6,
                                   max_order=None, trace=True)

        predict_dict[col[:-6]] = auto.predict(n_periods=PERIODS)
    except ValueError:
        print(f'Could not process {col}')
        unforecast.append(col)


exists = os.path.isfile(cwd + 'predict.pkl')

if exists:
    pass
else:
    file = open(cwd + 'predict.pkl', 'wb')
    pickle.dump(predict_dict, file)
