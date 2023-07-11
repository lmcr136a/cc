import requests
import json

BASE_URL = 'https://api.binance.com'
TIMEFRAME = '4h'
EMA_PERIODS = [50, 200]
symbols = []
candles = {}
pricers = {}
ema_values = {}
resp = requests.get(BASE_URL+'/api/v1/ticker/allBookTickers')
tickers_list = json.loads(resp.content)
for ticker in tickers_list:
    if str(ticker['symbol'][-4:]) == 'USDT':
        symbols.append(ticker['symbol'][:-4]+"/USDT")
with open("syms.txt", 'w') as f:
    f.write(str(symbols))