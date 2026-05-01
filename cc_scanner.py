#!/usr/bin/env python3
"""CC Sell Scanner — shows which tickers to sell covered calls on right now.
Uses multi-RSI combos (RSI5+RSI14) from backtested research.
Run: python3 ~/trading_bot/cc_scanner.py
"""
import sys, os
sys.path.insert(0, '/opt/jarvis-utils/lib')
from jarvis_utils.secrets import get
import requests
import yfinance as yf
import numpy as np
from datetime import datetime

API_KEY = get('Alpaca','api_key_id',user='a4dc8459-608d-49f5-943e-e5e105ed5207')
SECRET = get('Alpaca','secret_key',user='a4dc8459-608d-49f5-943e-e5e105ed5207')
HEADERS = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET}

TICKERS = ['WPM', 'GDX', 'KGC', 'PAAS', 'EGO', 'AG', 'GDXJ', 'SIL', 'HL', 'CDE', 'SVM', 'BTG', 'COPX']

# Per-ticker CC sell conditions (from multi-RSI backtest Apr 9)
CC_CONDITIONS = {
    'WPM': (80, 70),   # RSI5>80 + RSI14>70 — Sharpe 1.2, 94% WR
    'GDX': (70, 60),   # RSI5>70 + RSI14>60 — Sharpe 0.8, 93% WR
    'KGC': (80, 70),   # RSI5>80 + RSI14>70 — only profitable combo
}
DEFAULT_CC = (70, 60)  # RSI5>70 + RSI14>60

def rsi(prices, period):
    delta = np.diff(prices)
    vals = [50.0] * min(period, len(delta))
    for i in range(period, len(delta)):
        gains = np.where(delta[i-period:i] > 0, delta[i-period:i], 0)
        losses = np.where(delta[i-period:i] < 0, -delta[i-period:i], 0)
        ag, al = np.mean(gains), np.mean(losses)
        vals.append(100.0 if al == 0 else 100 - 100/(1+ag/al))
    return vals

def get_live(sym):
    try:
        r = requests.get(f'https://data.alpaca.markets/v2/stocks/{sym}/trades/latest',
                         headers=HEADERS, params={'feed': 'iex'}, timeout=5)
        return r.json().get('trade', {}).get('p', None)
    except:
        return None

def scan():
    print(f"=== CC SELL SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M')} ===\n")
    
    results = []
    for sym in TICKERS:
        try:
            h = yf.Ticker(sym).history(period='30d')
            closes = h['Close'].values.tolist()
            
            live = get_live(sym)
            if live and live > 0:
                closes.append(live)
            
            c = np.array(closes)
            r2 = rsi(c, 2)[-1]
            r5 = rsi(c, 5)[-1]
            r14 = rsi(c, 14)[-1]
            price = c[-1]
            
            r5_thresh, r14_thresh = CC_CONDITIONS.get(sym, DEFAULT_CC)
            sell = r5 > r5_thresh and r14 > r14_thresh
            
            results.append((sym, price, r2, r5, r14, sell, r5_thresh, r14_thresh))
        except Exception as e:
            print(f"  {sym}: error — {e}")
    
    # Sort: sells first, then by RSI14 desc
    results.sort(key=lambda x: (0 if x[5] else 1, -x[4]))
    
    print(f"{'Ticker':<6} {'Price':>8} {'RSI2':>5} {'RSI5':>5} {'RSI14':>5} {'Signal':>8} {'Need'}")
    for sym, price, r2, r5, r14, sell, r5t, r14t in results:
        signal = '🔥 SELL' if sell else '  wait'
        need = f"R5>{r5t} R14>{r14t}"
        print(f"{sym:<6} ${price:>7.2f} {r2:>5.0f} {r5:>5.0f} {r14:>5.0f} {signal}  {need}")
    
    sells = [r for r in results if r[5]]
    print(f"\n{len(sells)} tickers ready. Close target: 25% of premium (GTC buy-to-close).")

if __name__ == '__main__':
    scan()
