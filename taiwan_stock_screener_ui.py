"""
台股即時/盤後高勝率選股程式 - 視覺化 GUI 版本
作者：資深金融軟體工程師
"""

import os
import sys

# 解決 PyInstaller --noconsole 模式下 sys.stdout/sys.stderr 為 None 導致 loguru (FinMind) 崩潰之問題
class DummyStream:
    def write(self, data):
        pass
    def flush(self):
        pass
    def isatty(self):
        return False

if sys.stdout is None:
    sys.stdout = DummyStream()
if sys.stderr is None:
    sys.stderr = DummyStream()

import time
import datetime
import threading
import queue
import pandas as pd
import numpy as np
import yfinance as yf
from FinMind.data import DataLoader

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# 解決 Windows 終端機 UTF-8 輸出與 Emoji 顯示之編碼問題
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

class TaiwanStockScreenerEngine:
    def __init__(self, api_token: str = None, log_callback=None, progress_callback=None):
        """
        選股計算引擎
        :param api_token: FinMind API Token
        :param log_callback: 日誌回呼函數，用來更新 UI 中的日誌文字
        :param progress_callback: 進度條百分比回呼函數 (0-100)
        """
        self.api = DataLoader()
        if api_token:
            self.api.login_by_token(api_token)
            self.log("【系統】已成功登入 FinMind API 帳號。")
        else:
            self.log("【系統】未提供 FinMind API 憑證，將以免費模式執行 (注意 API 次數限制)。")
            
        self.log_callback = log_callback
        self.progress_callback = progress_callback

    def log(self, message: str):
        print(message)
        if self.log_callback:
            self.log_callback(message)

    def set_progress(self, percent: float):
        if self.progress_callback:
            self.progress_callback(percent)

    def fetch_ordinary_stocks(self) -> pd.DataFrame:
        self.log("【步驟 1】正在獲取台股上市/上櫃普通股清單...")
        self.set_progress(5)
        try:
            df_info = self.api.taiwan_stock_info()
            exclude_keywords = 'ETF|ETN|Wd|存託憑證|受益證券|特別股|債券|指數|wd|Wd'
            df_ordinary = df_info[
                (df_info['type'].isin(['twse', 'tpex'])) &
                (df_info['stock_id'].str.len() == 4) &
                (~df_info['industry_category'].str.contains(exclude_keywords, na=False, case=False))
            ].copy()
            
            df_ordinary['yf_ticker'] = df_ordinary.apply(
                lambda row: f"{row['stock_id']}.TW" if row['type'] == 'twse' else f"{row['stock_id']}.TWO",
                axis=1
            )
            self.log(f"【成功】已成功篩選出 {len(df_ordinary)} 檔台股普通股進行分析。")
            self.set_progress(10)
            return df_ordinary
        except Exception as e:
            self.log(f"【錯誤】獲取股票清單失敗: {e}")
            raise e

    def download_price_data(self, tickers: list, days_back: int = 60) -> pd.DataFrame:
        self.log(f"【步驟 2】開始從 Yahoo Finance 下載近 {days_back} 天的歷史股價資料...")
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        chunk_size = 300
        ticker_chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
        
        all_dfs = []
        total_chunks = len(ticker_chunks)
        
        for idx, chunk in enumerate(ticker_chunks):
            self.log(f"  -> 下載進度：第 {idx + 1}/{total_chunks} 批 (共 {len(chunk)} 檔)...")
            try:
                df_chunk = yf.download(chunk, start=start_date, end=end_date, group_by='ticker', progress=False)
                if not df_chunk.empty:
                    all_dfs.append(df_chunk)
                time.sleep(0.5)
            except Exception as e:
                self.log(f"  【警告】批次下載失敗，跳過該批次: {e}")
            
            # 下載進度從 10% 分布到 50%
            current_progress = 10 + int((idx + 1) / total_chunks * 40)
            self.set_progress(current_progress)
                
        if not all_dfs:
            self.log("【錯誤】未下載到任何股價資料！")
            raise Exception("未下載到任何股價資料")
            
        self.log("【成功】所有批次下載完成，進行資料合併與整理...")
        df_combined = pd.concat(all_dfs, axis=1)
        self.set_progress(50)
        return df_combined

    def run_technical_screening(self, df_prices: pd.DataFrame, df_ordinary: pd.DataFrame) -> list:
        self.log("【步驟 3】開始進行技術面策略篩選...")
        passed_tech = []
        available_tickers = df_prices.columns.levels[0].unique()
        total_tickers = len(available_tickers)
        
        for idx, ticker in enumerate(available_tickers):
            try:
                df_single = df_prices[ticker].dropna(subset=['Close'])
                if len(df_single) < 20:
                    continue
                
                close_series = df_single['Close']
                volume_series = df_single['Volume']
                
                # 條件 D（濾網 - 排除妖股與殭屍股）
                latest_close = close_series.iloc[-1]
                mean_vol_20 = volume_series.tail(20).mean()
                
                if latest_close <= 10 or mean_vol_20 <= 1000000:
                    continue
                
                # 條件 A（技術面 - 強勢突破）
                high_20 = close_series.tail(20).max()
                is_price_breakout = (latest_close >= high_20)
                
                latest_volume = volume_series.iloc[-1]
                is_volume_breakout = (latest_volume > 1.5 * mean_vol_20)
                
                if not (is_price_breakout and is_volume_breakout):
                    continue
                
                # 條件 B（技術面 - 均線多頭排列）
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
            except Exception:
                pass
            
            # 技術面篩選進度從 50% 分布到 60%
            if idx % 100 == 0:
                current_progress = 50 + int((idx + 1) / total_tickers * 10)
                self.set_progress(current_progress)
                
        self.log(f"【成功】技術面篩選完畢。共有 {len(passed_tech)} 檔股票符合技術面條件。")
        self.set_progress(60)
        return passed_tech

    def run_chip_screening(self, tech_candidates: list) -> list:
        self.log("【步驟 4】開始從 FinMind 獲取法人籌碼資料，進行籌碼面條件篩選...")
        final_selected = []
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=30)).strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        total_candidates = len(tech_candidates)
        
        for idx, s in enumerate(tech_candidates):
            stock_id = s['stock_id']
            self.log(f"  -> 分析中 ({idx+1}/{total_candidates}): {stock_id} {s['stock_name']}...")
            
            try:
                df_inst = self.api.taiwan_stock_institutional_investors(
                    stock_id=stock_id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df_inst.empty:
                    continue
                
                df_foreign = df_inst[df_inst['name'] == 'Foreign_Investor']
                df_trust = df_inst[df_inst['name'] == 'Investment_Trust']
                
                df_foreign_grouped = df_foreign.groupby('date')[['buy', 'sell']].sum()
                df_trust_grouped = df_trust.groupby('date')[['buy', 'sell']].sum()
                
                df_foreign_net = df_foreign_grouped['buy'] - df_foreign_grouped['sell']
                df_trust_net = df_trust_grouped['buy'] - df_trust_grouped['sell']
                
                df_merged = pd.DataFrame({
                    'foreign_net': df_foreign_net,
                    'trust_net': df_trust_net
                }).dropna().sort_index()
                
                if len(df_merged) < 3:
                    continue
                
                # 條件 C
                df_last3 = df_merged.tail(3).copy()
                df_last3['sync_buy'] = (df_last3['foreign_net'] > 0) & (df_last3['trust_net'] > 0)
                sync_days_count = df_last3['sync_buy'].sum()
                
                if sync_days_count >= 2:
                    latest_foreign_net_lots = round(df_last3['foreign_net'].iloc[-1] / 1000, 1)
                    latest_trust_net_lots = round(df_last3['trust_net'].iloc[-1] / 1000, 1)
                    
                    s['foreign_net_buy_lots'] = latest_foreign_net_lots
                    s['trust_net_buy_lots'] = latest_trust_net_lots
                    s['sync_days'] = int(sync_days_count)
                    final_selected.append(s)
                    self.log(f"    ★ 符合條件！過去 3 天有 {sync_days_count} 天同步買超。當日外資買超 {latest_foreign_net_lots} 張，投信買超 {latest_trust_net_lots} 張。")
            except Exception as e:
                self.log(f"    【警告】分析 {stock_id} 籌碼時出錯: {e}")
            
            # 籌碼篩選進度從 60% 分布到 95%
            current_progress = 60 + int((idx + 1) / total_candidates * 35)
            self.set_progress(current_progress)
                
        self.log(f"【成功】籌碼面篩選完畢。共有 {len(final_selected)} 檔股票符合所有選股條件！")
        self.set_progress(100)
        return final_selected


class TaiwanStockScreenerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("台股高勝率選股系統 v1.0")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        
        self.df_result = pd.DataFrame()
        self.screener_thread = None
        self.log_queue = queue.Queue()
        
        self.setup_ui()
        self.root.after(100, self.process_log_queue)

    def setup_ui(self):
        # 設置現代視窗樣式
        self.style = ttk.Style()
        self.style.theme_use('vista' if 'vista' in self.style.theme_names() else 'clam')
        
        # 設置顏色與字型風格
        self.font_title = ("Microsoft JhengHei", 18, "bold")
        self.font_subtitle = ("Microsoft JhengHei", 10)
        self.font_text = ("Microsoft JhengHei", 10)
        self.font_btn = ("Microsoft JhengHei", 10, "bold")
        
        # 自訂 ttk 樣式
        self.style.configure("TProgressbar", thickness=15)
        self.style.configure("Accent.TButton", font=self.font_btn, foreground="white", background="#007ACC")
        self.style.configure("Action.TButton", font=self.font_btn)
        
        # 1. 頂部標題區 (Header)
        header_frame = tk.Frame(self.root, bg="#2C3E50", height=80)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame, 
            text="台股即時/盤後高勝率選股系統", 
            font=self.font_title, 
            fg="#ECF0F1", 
            bg="#2C3E50"
        )
        title_label.pack(anchor=tk.W, padx=20, pady=(15, 2))
        
        subtitle_label = tk.Label(
            header_frame, 
            text="技術面 (突破新高 + 均線多頭) ＋ 籌碼面 (外資投信同步買超) 雙重確認策略", 
            font=self.font_subtitle, 
            fg="#BDC3C7", 
            bg="#2C3E50"
        )
        subtitle_label.pack(anchor=tk.W, padx=20, pady=(0, 10))

        # 2. 控制面板區 (Control Panel)
        ctrl_frame = ttk.LabelFrame(self.root, text=" 系統控制與狀態 ")
        ctrl_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # 按鈕容器
        btn_container = ttk.Frame(ctrl_frame)
        btn_container.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_start = tk.Button(
            btn_container, 
            text="🚀 開始篩選股票", 
            font=self.font_btn, 
            bg="#2ECC71", 
            fg="white", 
            activebackground="#27AE60", 
            activeforeground="white",
            relief=tk.FLAT, 
            padx=15, 
            pady=6,
            command=self.start_screening
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))
        
        self.btn_excel = tk.Button(
            btn_container, 
            text="📥 匯出 Excel (.xlsx)", 
            font=self.font_btn, 
            bg="#3498DB", 
            fg="white", 
            activebackground="#2980B9", 
            activeforeground="white",
            relief=tk.FLAT, 
            padx=15, 
            pady=6,
            state=tk.DISABLED,
            command=self.export_excel
        )
        self.btn_excel.pack(side=tk.LEFT, padx=10)
        
        self.btn_csv = tk.Button(
            btn_container, 
            text="📥 匯出 CSV", 
            font=self.font_btn, 
            bg="#95A5A6", 
            fg="white", 
            activebackground="#7F8C8D", 
            activeforeground="white",
            relief=tk.FLAT, 
            padx=15, 
            pady=6,
            state=tk.DISABLED,
            command=self.export_csv
        )
        self.btn_csv.pack(side=tk.LEFT, padx=10)
        
        # 進度條與狀態文字
        progress_container = ttk.Frame(ctrl_frame)
        progress_container.pack(fill=tk.X, padx=10, pady=(5, 10))
        
        self.lbl_status = ttk.Label(progress_container, text="系統狀態：準備就緒，點擊按鈕開始選股", font=self.font_subtitle)
        self.lbl_status.pack(anchor=tk.W, pady=(0, 5))
        
        self.progressbar = ttk.Progressbar(progress_container, mode='determinate')
        self.progressbar.pack(fill=tk.X)

        # 3. 主內容區 (Main Workspace: 左右分割)
        main_pane = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        # 下半部表格區
        table_frame = ttk.LabelFrame(main_pane, text=" 符合黃金交叉策略之個股清單 ")
        main_pane.add(table_frame, weight=3)
        
        # 設置樹狀表 (Treeview)
        columns = ("stock_id", "stock_name", "date", "close", "daily_return", "volume", "foreign_buy", "trust_buy")
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        self.tree.heading("stock_id", text="股票代號")
        self.tree.heading("stock_name", text="股票名稱")
        self.tree.heading("date", text="資料日期")
        self.tree.heading("close", text="當日收盤價")
        self.tree.heading("daily_return", text="當日漲跌幅(%)")
        self.tree.heading("volume", text="成交量(張)")
        self.tree.heading("foreign_buy", text="外資買超(張)")
        self.tree.heading("trust_buy", text="投信買超(張)")
        
        # 設定欄寬與置中
        for col in columns:
            self.tree.column(col, anchor=tk.CENTER, width=110)
            
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        scrollbar.pack(fill=tk.Y, side=tk.RIGHT)
        
        # 上半部日誌區
        log_frame = ttk.LabelFrame(main_pane, text=" 選股進度與日誌 ")
        main_pane.add(log_frame, weight=2)
        
        self.log_text = ScrolledText(log_frame, height=8, font=("Consolas", 9), bg="#1E1E1E", fg="#F1F1F1")
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.insert(tk.END, "★ 歡迎使用本選股程式！\n★ 本策略將自動：過濾 ETF -> 抓取 3000+ 檔台股日K線 -> 篩選強勢突破與多頭排列股 -> 下載三大法人買賣超資料 -> 找出外資投信鎖碼股。\n★ 請點選上方「開始篩選股票」啟動流程。\n\n")
        self.log_text.configure(state=tk.DISABLED)

    def append_log_safe(self, message: str):
        self.log_queue.put(message)

    def process_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, f"{msg}\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
                
                # 自動根據日誌文字更新狀態標籤
                if "步驟 1" in msg:
                    self.lbl_status.config(text="系統狀態：正在獲取台股上市/上櫃普通股清單...")
                elif "步驟 2" in msg:
                    self.lbl_status.config(text="系統狀態：正在從 Yahoo Finance 批量下載歷史日K價格...")
                elif "步驟 3" in msg:
                    self.lbl_status.config(text="系統狀態：正在進行技術面分析 (計算均線、波動與流動性)...")
                elif "步驟 4" in msg:
                    self.lbl_status.config(text="系統狀態：正在查詢 FinMind 法人籌碼鎖碼狀況...")
                elif "符合所有選股條件" in msg:
                    self.lbl_status.config(text="系統狀態：選股篩選完成！")
                
                self.log_queue.task_done()
        except queue.Empty:
            pass
        self.root.after(100, self.process_log_queue)

    def update_progress(self, percent: float):
        # 使用 after 確保在 GUI 線程更新
        self.root.after(0, lambda: self.progressbar.configure(value=percent))

    def start_screening(self):
        if self.screener_thread and self.screener_thread.is_alive():
            messagebox.showwarning("執行中", "選股流程已在運行中，請耐心等候！")
            return
            
        self.btn_start.config(state=tk.DISABLED, bg="#95A5A6")
        self.btn_excel.config(state=tk.DISABLED, bg="#95A5A6")
        self.btn_csv.config(state=tk.DISABLED, bg="#95A5A6")
        
        # 清空表格與日誌
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
        self.progressbar.configure(value=0)
        self.lbl_status.config(text="系統狀態：啟動選股引擎中...")
        
        # 啟動後台線程執行計算
        self.screener_thread = threading.Thread(target=self.run_screener_flow, daemon=True)
        self.screener_thread.start()

    def run_screener_flow(self):
        start_time = time.time()
        engine = TaiwanStockScreenerEngine(
            log_callback=self.append_log_safe, 
            progress_callback=self.update_progress
        )
        
        try:
            # 1. 獲取普通股
            df_ordinary = engine.fetch_ordinary_stocks()
            tickers = df_ordinary['yf_ticker'].tolist()
            
            # 2. 下載股價資料
            df_prices = engine.download_price_data(tickers)
            
            # 3. 技術面篩選
            tech_candidates = engine.run_technical_screening(df_prices, df_ordinary)
            
            if not tech_candidates:
                engine.log("\n【結果】沒有任何股票符合技術面條件，選股結束。")
                self.root.after(0, lambda: messagebox.showinfo("完成", "本次篩選未發現符合技術面條件的個股。"))
                self.root.after(0, self.on_screening_finished)
                return
                
            # 4. 籌碼面篩選
            final_candidates = engine.run_chip_screening(tech_candidates)
            
            # 5. 整理輸出
            if final_candidates:
                df = pd.DataFrame(final_candidates)
                df = df[[
                    'stock_id', 'stock_name', 'date', 'close', 'daily_return', 
                    'volume_lots', 'foreign_net_buy_lots', 'trust_net_buy_lots'
                ]]
                df.columns = [
                    '股票代號', '股票名稱', '資料日期', '當日收盤價', '當日漲跌幅(%)', 
                    '成交量(張)', '外資買超(張)', '投信買超(張)'
                ]
                df = df.sort_values(by='當日漲跌幅(%)', ascending=False)
                self.df_result = df
                
                # 自動存檔至 selected_stocks.csv
                df.to_csv("selected_stocks.csv", index=False, encoding='utf-8-sig')
                engine.log(f"\n【存檔】選股清單已成功儲存至工作目錄下的 [selected_stocks.csv]。")
                
                # 將資料填充至 Treeview 中
                self.root.after(0, lambda: self.populate_table(df))
                elapsed = time.time() - start_time
                engine.log(f"【總耗時】整個選股流程共耗時：{elapsed:.2f} 秒。")
                self.root.after(0, lambda: messagebox.showinfo("成功", f"選股完成！共有 {len(df)} 檔股票符合所有條件。"))
            else:
                engine.log("\n【結果】無符合所有條件之個股。")
                self.root.after(0, lambda: messagebox.showinfo("完成", "本次篩選未發現符合所有技術與籌碼條件的個股。"))
                
        except Exception as e:
            engine.log(f"\n【錯誤】執行選股時遭遇嚴重錯誤: {e}")
            self.root.after(0, lambda: messagebox.showerror("錯誤", f"選股流程中斷: {e}"))
            
        self.root.after(0, self.on_screening_finished)

    def populate_table(self, df: pd.DataFrame):
        for idx, row in df.iterrows():
            self.tree.insert("", tk.END, values=(
                row['股票代號'],
                row['股票名稱'],
                row['資料日期'],
                f"{row['當日收盤價']:.2f}",
                f"{row['當日漲跌幅(%)']:.2f}",
                f"{row['成交量(張)']:,}",
                f"{row['外資買超(張)']:,}",
                f"{row['投信買超(張)']:,}"
            ))

    def on_screening_finished(self):
        self.btn_start.config(state=tk.NORMAL, bg="#2ECC71")
        if not self.df_result.empty:
            self.btn_excel.config(state=tk.NORMAL, bg="#3498DB")
            self.btn_csv.config(state=tk.NORMAL, bg="#95A5A6")
        self.lbl_status.config(text=f"系統狀態：分析結束。時間: {datetime.datetime.now().strftime('%H:%M:%S')}")

    def export_excel(self):
        if self.df_result.empty:
            messagebox.showwarning("警告", "目前無選股結果資料可供匯出。")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel Files", "*.xlsx")],
            title="選擇儲存 Excel 檔案位置",
            initialfile=f"選股報告_{datetime.date.today().strftime('%Y%m%d')}.xlsx"
        )
        
        if not file_path:
            return
            
        try:
            # 匯出至 Excel 格式
            self.df_result.to_excel(file_path, index=False)
            messagebox.showinfo("成功", f"選股報告已成功匯出至：\n{file_path}")
        except Exception as e:
            messagebox.showerror("錯誤", f"匯出 Excel 失敗: {e}")

    def export_csv(self):
        if self.df_result.empty:
            messagebox.showwarning("警告", "目前無選股結果資料可供匯出。")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="選擇儲存 CSV 檔案位置",
            initialfile=f"選股報告_{datetime.date.today().strftime('%Y%m%d')}.csv"
        )
        
        if not file_path:
            return
            
        try:
            # 使用 utf-8-sig 防止 Excel 亂碼
            self.df_result.to_csv(file_path, index=False, encoding='utf-8-sig')
            messagebox.showinfo("成功", f"選股報告已成功匯出至：\n{file_path}")
        except Exception as e:
            messagebox.showerror("錯誤", f"匯出 CSV 失敗: {e}")


def main():
    root = tk.Tk()
    app = TaiwanStockScreenerUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
