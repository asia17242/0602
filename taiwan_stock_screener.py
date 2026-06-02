"""
台股即時/盤後高勝率選股程式
作者：資深金融軟體工程師
核心策略：籌碼面與技術面黃金交叉策略 (強勢突破 + 均線多頭 + 法人鎖碼 + 流動性濾網)
"""

import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from FinMind.data import DataLoader

# 解決 Windows 終端機 UTF-8 輸出與 Emoji 顯示之編碼問題
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass


class TaiwanStockScreener:
    def __init__(self, api_token: str = None):
        """
        初始化選股程式
        :param api_token: FinMind API 憑證 (若無則使用免費流量限制版)
        """
        self.api = DataLoader()
        if api_token:
            self.api.login_by_token(api_token)
            print("【系統】已成功登入 FinMind API 帳號。")
        else:
            print("【系統】未提供 FinMind API 憑證，將以免費模式執行 (注意 API 次數限制)。")

    def fetch_ordinary_stocks(self) -> pd.DataFrame:
        """
        從 FinMind 獲取台股所有股票清單，並過濾掉 ETF、warrants、存託憑證等，僅保留普通股
        """
        print("【步驟 1】正在獲取台股上市/上櫃普通股清單...")
        try:
            df_info = self.api.taiwan_stock_info()
            # 過濾條件：
            # 1. 交易類型為 twse (上市) 或 tpex (上櫃)
            # 2. 股票代碼為 4 位數 (排除權證、六位數ETF等)
            # 3. 排除產業類別中包含 ETF、ETN、存託憑證、受益證券、特別股、債券等非普通股
            exclude_keywords = 'ETF|ETN|Wd|存託憑證|受益證券|特別股|債券|指數|wd|Wd'
            df_ordinary = df_info[
                (df_info['type'].isin(['twse', 'tpex'])) &
                (df_info['stock_id'].str.len() == 4) &
                (~df_info['industry_category'].str.contains(exclude_keywords, na=False, case=False))
            ].copy()
            
            # 對應 yfinance 股票代號格式 (上市為 .TW, 上櫃為 .TWO)
            df_ordinary['yf_ticker'] = df_ordinary.apply(
                lambda row: f"{row['stock_id']}.TW" if row['type'] == 'twse' else f"{row['stock_id']}.TWO",
                axis=1
            )
            print(f"【成功】已成功篩選出 {len(df_ordinary)} 檔台股上市/上櫃普通股進行分析。")
            return df_ordinary
        except Exception as e:
            print(f"【錯誤】獲取股票清單失敗: {e}")
            sys.exit(1)

    def download_price_data(self, tickers: list, days_back: int = 60) -> pd.DataFrame:
        """
        分批下載歷史日K線資料
        :param tickers: 股票代號清單 (yfinance 格式)
        :param days_back: 回溯天數，預設為 60 天 (足夠計算 20 日均線及新高)
        """
        print(f"【步驟 2】開始從 Yahoo Finance 下載近 {days_back} 天的歷史股價資料...")
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        # 由於股票數量多，分批下載 (每批 300 檔) 以防請求過長或逾時
        chunk_size = 300
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
        
        all_dfs = []
        for idx, chunk in enumerate(ticker_chunks):
            print(f"  -> 下載進度：第 {idx + 1}/{len(ticker_chunks)} 批 (共 {len(chunk)} 檔)...")
            try:
                # yfinance 批量下載
                df_chunk = yf.download(chunk, start=start_date, end=end_date, group_by='ticker', progress=False)
                if not df_chunk.empty:
                    all_dfs.append(df_chunk)
                # 稍微延遲避免請求過頻
                time.sleep(1)
            except Exception as e:
                print(f"  【警告】批次下載失敗，跳過該批次: {e}")
                
        if not all_dfs:
            print("【錯誤】未下載到任何股價資料！")
            sys.exit(1)
            
        print("【成功】所有批次下載完成，進行資料合併與整理...")
        df_combined = pd.concat(all_dfs, axis=1)
        return df_combined

    def run_technical_screening(self, df_prices: pd.DataFrame, df_ordinary: pd.DataFrame) -> list:
        """
        執行技術面篩選 (條件 A, B, D)
        """
        print("【步驟 3】開始進行技術面策略篩選...")
        passed_tech = []
        available_tickers = df_prices.columns.levels[0].unique()
        
        for idx, ticker in enumerate(available_tickers):
            try:
                # 提取個股歷史價格並剔除空值
                df_single = df_prices[ticker].dropna(subset=['Close'])
                if len(df_single) < 20:
                    continue
                
                close_series = df_single['Close']
                volume_series = df_single['Volume']
                
                # 1. 條件 D（濾網 - 排除妖股與殭屍股）
                # 股價 > 10 元，且過去 20 天日平均成交量 > 1000 張 (1,000,000 股)
                latest_close = close_series.iloc[-1]
                mean_vol_20 = volume_series.tail(20).mean()
                
                if latest_close <= 10 or mean_vol_20 <= 1000000:
                    continue
                
                # 2. 條件 A（技術面 - 強勢突破）
                # 當日收盤價創下過去 20 天的新高，且當日成交量大於過去 20 天平均成交量的 1.5 倍
                high_20 = close_series.tail(20).max()
                is_price_breakout = (latest_close >= high_20)
                
                latest_volume = volume_series.iloc[-1]
                is_volume_breakout = (latest_volume > 1.5 * mean_vol_20)
                
                if not (is_price_breakout and is_volume_breakout):
                    continue
                
                # 3. 條件 B（技術面 - 均線多頭排列）
                # 5日均線 > 10日均線 > 20日均線 (MA5 > MA10 > MA20)
                ma5 = close_series.rolling(5).mean().iloc[-1]
                ma10 = close_series.rolling(10).mean().iloc[-1]
                ma20 = close_series.rolling(20).mean().iloc[-1]
                
                if not (ma5 > ma10 > ma20):
                    continue
                
                # 計算當日漲跌幅 (%)
                prev_close = close_series.iloc[-2]
                daily_return = ((latest_close - prev_close) / prev_close) * 100
                
                # 獲取中文名稱及日期
                stock_id = ticker.split('.')[0]
                stock_info_row = df_ordinary[df_ordinary['stock_id'] == stock_id]
                stock_name = stock_info_row.iloc[0]['stock_name'] if not stock_info_row.empty else "未知"
                data_date = df_single.index[-1].strftime('%Y-%m-%d')
                
                passed_tech.append({
                    'stock_id': stock_id,
                    'stock_name': stock_name,
                    'date': data_date,
                    'close': round(latest_close, 2),
                    'daily_return': round(daily_return, 2),
                    'volume_lots': round(latest_volume / 1000, 1),
                    'mean_vol_20_lots': round(mean_vol_20 / 1000, 1)
                })
            except Exception as e:
                # 異常處理：若單一股票計算出錯，跳過該股票，避免影響整個選股程式執行
                pass
                
        print(f"【成功】技術面篩選完畢。共有 {len(passed_tech)} 檔股票符合技術面條件 (強勢突破 + 多頭排列 + 流動性濾網)。")
        return passed_tech

    def run_chip_screening(self, tech_candidates: list) -> list:
        """
        執行籌碼面篩選 (條件 C)，僅針對技術面符合的候選股票
        """
        print("【步驟 4】開始從 FinMind 獲取法人籌碼資料，進行籌碼面條件篩選...")
        final_selected = []
        today = datetime.date.today()
        # 抓取過去 30 天的法人買賣超資料，確保即使遇到長假也能有完整的 3 個交易日資料
        start_date = (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        for idx, s in enumerate(tech_candidates):
            stock_id = s['stock_id']
            print(f"  -> 分析中 ({idx+1}/{len(tech_candidates)}): {stock_id} {s['stock_name']}...")
            
            try:
                # 獲取三大法人買賣超資料
                df_inst = self.api.taiwan_stock_institutional_investors(
                    stock_id=stock_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df_inst.empty:
                    continue
                
                # 分別過濾外資與投信資料
                df_foreign = df_inst[df_inst['name'] == 'Foreign_Investor']
                df_trust = df_inst[df_inst['name'] == 'Investment_Trust']
                
                # 分組計算每日買賣淨額
                df_foreign_grouped = df_foreign.groupby('date')[['buy', 'sell']].sum()
                df_trust_grouped = df_trust.groupby('date')[['buy', 'sell']].sum()
                
                df_foreign_net = df_foreign_grouped['buy'] - df_foreign_grouped['sell']
                df_trust_net = df_trust_grouped['buy'] - df_trust_grouped['sell']
                
                # 合併外資與投信日淨額
                df_merged = pd.DataFrame({
                    'foreign_net': df_foreign_net,
                    'trust_net': df_trust_net
                }).dropna().sort_index()
                
                # 確保至少有 3 個交易日的法人資料
                if len(df_merged) < 3:
                    continue
                
                # 條件 C（籌碼面 - 法人鎖碼）
                # 外資與投信在過去 3 個交易日中，至少有 2 天呈現「同步買超」
                df_last3 = df_merged.tail(3).copy()
                df_last3['sync_buy'] = (df_last3['foreign_net'] > 0) & (df_last3['trust_net'] > 0)
                sync_days_count = df_last3['sync_buy'].sum()
                
                if sync_days_count >= 2:
                    # 抓取最後一日的外資與投信買超張數 (單位轉換為張, 股數/1000)
                    latest_foreign_net_lots = round(df_last3['foreign_net'].iloc[-1] / 1000, 1)
                    latest_trust_net_lots = round(df_last3['trust_net'].iloc[-1] / 1000, 1)
                    
                    s['foreign_net_buy_lots'] = latest_foreign_net_lots
                    s['trust_net_buy_lots'] = latest_trust_net_lots
                    s['sync_days'] = int(sync_days_count)
                    final_selected.append(s)
                    print(f"    ★ 符合條件！過去 3 天有 {sync_days_count} 天同步買超。當日外資買超 {latest_foreign_net_lots} 張，投信買超 {latest_trust_net_lots} 張。")
            except Exception as e:
                # 異常處理：防範單一股票 API 請求錯誤或資料清洗出錯
                print(f"    【警告】分析 {stock_id} 籌碼時出錯: {e}")
                
        print(f"【成功】籌碼面篩選完畢。共有 {len(final_selected)} 檔股票符合所有選股條件！")
        return final_selected

    def execute_screening_flow(self) -> pd.DataFrame:
        """
        執行完整的選股流程
        """
        start_time = time.time()
        
        # 1. 獲取股票清單
        df_ordinary = self.fetch_ordinary_stocks()
        tickers = df_ordinary['yf_ticker'].tolist()
        
        # 2. 下載股價
        df_prices = self.download_price_data(tickers)
        
        # 3. 技術面篩選
        tech_candidates = self.run_technical_screening(df_prices, df_ordinary)
        
        if not tech_candidates:
            print("\n【結果】沒有任何股票符合技術面條件，選股結束。")
            return pd.DataFrame()
            
        # 4. 籌碼面篩選
        final_candidates = self.run_chip_screening(tech_candidates)
        
        print("\n" + "="*50 + "\n【選股結果報告】\n" + "="*50)
        
        if not final_candidates:
            print("本日無符合篩選條件之股票。")
            return pd.DataFrame()
            
        # 5. 整理輸出結果
        df_result = pd.DataFrame(final_candidates)
        df_result = df_result[[
            'stock_id', 'stock_name', 'date', 'close', 'daily_return', 
            'volume_lots', 'foreign_net_buy_lots', 'trust_net_buy_lots'
        ]]
        df_result.columns = [
            '股票代號', '股票名稱', '資料日期', '當日收盤價', '當日漲跌幅(%)', 
            '成交量(張)', '外資買超(張)', '投信買超(張)'
        ]
        
        # 排序：優先按漲跌幅排序
        df_result = df_result.sort_values(by='當日漲跌幅(%)', ascending=False)
        
        # 列印到終端機
        print(df_result.to_string(index=False))
        
        # 儲存為 CSV
        output_file = "selected_stocks.csv"
        # 使用 utf-8-sig 格式編碼，確保 Windows Excel 開啟中文不會亂碼
        df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n【存檔】選股清單已成功儲存至當前目錄的 [{output_file}]。")
        print(f"【總耗時】整個選股流程共耗時：{time.time() - start_time:.2f} 秒。")
        
        return df_result

if __name__ == '__main__':
    # 執行選股
    screener = TaiwanStockScreener()
    screener.execute_screening_flow()
