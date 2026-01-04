# 🛡️ FameKeeper: 企業聲譽智慧監測系統

**FameKeeper** 是一款專為企業、品牌及個人設計的智慧聲譽監測工具。透過整合主流社群媒體與搜尋引擎，並結合 LLM（大語言模型）進行情感分析與議題摘要，幫助使用者第一時間掌握輿情趨勢與潛在危機。

---

## ✨ 核心特色

* **多模型驅動**：支援 **Google Gemini 2.5 Flash** 與 **OpenAI GPT-4.1 Mini** 雙模型切換。
* **全方位監測**：整合 Google News、DuckDuckGo、Reddit 以及 YouTube 評論。
* **地區深度優化**：支援「台灣 (TW)」與「國際 (Global)」切換，自動調整搜尋語系與地區參數。
* **AI 自動化報告**：不僅進行正負面判讀，更會自動生成包含「趨勢、主要議題、風險建議」的摘要報告。
* **數據可視化**：內建數據儀表板，即時呈現情感分佈、平台佔比與互動指標。

---

## 🚀 快速上手

### 1. 環境準備

建議使用 Python 3.9 或更高版本。

```bash
# 複製專案
git clone https://github.com/your-username/FameKeeper.git
cd FameKeeper

# 安裝必要套件
pip install streamlit pandas sqlite3 google-genai feedparser duckduckgo_search

```

### 2. 選配套件 (依需求安裝)

若要啟用 Reddit 或 YouTube 監測，請安裝以下套件：

```bash
# 支援 Reddit
pip install praw

# 支援 YouTube
pip install google-api-python-client

# 支援 OpenAI
pip install openai

```

### 3. 啟動應用程式

```bash
streamlit run app.py

```

---

## 🛠️ 配置說明

啟動系統後，請在側邊欄設定相關 API Key：

| 類別 | 項目 | 說明 |
| --- | --- | --- |
| **AI 模型** | Gemini / OpenAI | 至少需填入一個 API Key 才能執行情感分析。 |
| **搜尋來源** | Google / DuckDuckGo | 免 API Key，開箱即用。 |
| **社群來源** | Reddit / YouTube | 需申請開發者帳號並取得 Client ID / API Key。 |

---

## 📊 系統架構

1. **DataFetcher**: 負責各平台的爬蟲與資料清理，支援地區化搜尋。
2. **UnifiedAIAnalyzer**: 統一封裝 Gemini 與 OpenAI API，負責 JSON 格式化情感標記。
3. **DatabaseManager**: 基於 SQLite 的本地存儲，記錄歷史監控數據。
4. **Streamlit UI**: 提供互動式搜尋、圖表展示與報告生成介面。

---

## 📝 版本資訊

* **版本**：1.1.0
* **更新日期**：2026-01-04
* **作者**：jkloip 陳信勇

---

## 📜 免責聲明

本工具僅供市場研究與輿情分析使用。使用者在使用爬蟲功能時，請務必遵守各平台的服務條款（ToS）及隱私權政策。

---
