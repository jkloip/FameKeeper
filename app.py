"""
企業聲譽智慧監測系統 (FameKeeper)
================================
整合多個資料源（Reddit、YouTube、Google News、DuckDuckGo）
支援多模型切換（Gemini AI / OpenAI）與地區篩選

作者：jkloip
版本：1.1.0 

"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import time
import random
import importlib.util
import json
from urllib.parse import quote  # 移至頂層引用

# === 核心套件（必裝） ===
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS  # type: ignore
from google import genai
import feedparser

# OpenAI 支援
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


# ============================================================================
# 第一部分：系統配置與初始化
# ============================================================================

class PackageChecker:
    """檢查可選套件是否已安裝"""
    
    @staticmethod
    def is_installed(package_name: str) -> bool:
        return importlib.util.find_spec(package_name) is not None


class Config:
    """應用程式配置管理"""
    
    # 檢查可選套件
    REDDIT_AVAILABLE = PackageChecker.is_installed('praw')
    YOUTUBE_AVAILABLE = PackageChecker.is_installed('googleapiclient')
    OPENAI_AVAILABLE = PackageChecker.is_installed('openai')
    
    # API Keys（將由 UI 輸入）
    GEMINI_API_KEY = ""
    OPENAI_API_KEY = ""
    
    REDDIT_CLIENT_ID = ""
    REDDIT_CLIENT_SECRET = ""
    REDDIT_USER_AGENT = "reputation_monitor/2.1"
    YOUTUBE_API_KEY = ""
    
    # 資料庫設定
    DB_NAME = "reputation.db"
    
    # 模型設定
    GEMINI_MODEL = "gemini-2.5-flash"
    OPENAI_MODEL = "gpt-4.1-mini" # 使用較經濟實惠的模型


# ============================================================================
# 第二部分：資料庫管理
# ============================================================================

class DatabaseManager:
    """資料庫操作管理類別"""
    
    def __init__(self, db_name: str = Config.DB_NAME):
        self.db_name = db_name
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mentions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    sentiment_score REAL,
                    sentiment_label TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    url TEXT,
                    author TEXT,
                    engagement INTEGER DEFAULT 0
                )
            ''')
            conn.commit()
    
    def save_mention(self, content: str, platform: str, score: float, 
                    label: str, url: str, author: str = "Unknown", 
                    engagement: int = 0):
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO mentions 
                (content, platform, sentiment_score, sentiment_label, 
                 timestamp, url, author, engagement) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (content, platform, score, label, datetime.now(), 
                  url, author, engagement))
            conn.commit()
    
    def get_recent_mentions(self, limit: int = 100) -> pd.DataFrame:
        with sqlite3.connect(self.db_name) as conn:
            query = "SELECT * FROM mentions ORDER BY timestamp DESC LIMIT ?"
            return pd.read_sql_query(query, conn, params=(limit,))
    
    def clear_all_data(self):
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("DELETE FROM mentions")
            conn.commit()


# ============================================================================
# 第三部分：AI 分析服務 (整合 Gemini 與 OpenAI)
# ============================================================================

class UnifiedAIAnalyzer:
    """整合 Gemini 與 OpenAI 的分析服務"""
    
    def __init__(self, provider: str, api_key: str):
        self.provider = provider
        self.api_key = api_key
        self.client = None
        
        if self.provider == 'gemini':
            self.client = genai.Client(api_key=api_key) if api_key else None
            self.model = Config.GEMINI_MODEL
        elif self.provider == 'openai':
            if Config.OPENAI_AVAILABLE and api_key:
                self.client = OpenAI(api_key=api_key)
                self.model = Config.OPENAI_MODEL
            else:
                self.client = None

    def analyze_sentiment(self, text: str) -> tuple:
        """分析文字情感"""
        if not self.client:
            return 0.0, "中立", "無法分析（未設定 API Key 或套件缺失）"
        
        prompt = self._build_sentiment_prompt(text[:500])
        
        try:
            if self.provider == 'gemini':
                return self._analyze_with_gemini(prompt)
            elif self.provider == 'openai':
                return self._analyze_with_openai(prompt)
        except Exception as e:
            error_str = str(e)
            if "quota" in error_str.lower() or "429" in error_str:
                return 0.0, "中立", "❗ API 配額已用盡"
            return 0.0, "中立", f"分析失敗: {error_str[:50]}"
        
        return 0.0, "中立", "未知錯誤"

    def _analyze_with_gemini(self, prompt: str) -> tuple:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        result = self._parse_json_response(response.text)
        return self._extract_result_tuple(result)

    def _analyze_with_openai(self, prompt: str) -> tuple:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        result_text = response.choices[0].message.content
        result = self._parse_json_response(result_text)
        return self._extract_result_tuple(result)

    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """生成整體報告"""
        if not self.client or df.empty:
            return "暫無數據可分析"
        
        # 準備資料
        sample_data = df.head(10)[['platform', 'sentiment_label', 'content']].to_dict('records')
        stats = self._calculate_statistics(df)
        prompt = self._build_summary_prompt(sample_data, stats)
        
        try:
            if self.provider == 'gemini':
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                return response.text
            elif self.provider == 'openai':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
        except Exception as e:
            return f"報告生成失敗: {str(e)[:100]}...\n請檢查 API Key 或配額。"

    # --- Helper Methods ---
    
    @staticmethod
    def _extract_result_tuple(result: dict) -> tuple:
        return (
            result.get("score", 0.0),
            result.get("label", "中立"),
            result.get("key_topic", "無法摘要")
        )

    @staticmethod
    def _build_sentiment_prompt(text: str) -> str:
        return f"""請分析以下文字的情感與關鍵議題：
文字內容："{text}"
請務必回傳標準 JSON 格式（不要有 markdown 標記），格式如下：
{{
  "score": -1.0 到 1.0 之間的浮點數,
  "label": "正面" 或 "負面" 或 "中立",
  "key_topic": "一句話摘要主要討論的議題"
}}"""
    
    @staticmethod
    def _build_summary_prompt(sample_data: list, stats: dict) -> str:
        return f"""你是企業、品牌與個人聲譽分析專家，根據以下社群媒體監測資料，生成一份專業的聲譽摘要報告（約 150 字）：
資料樣本：{sample_data}
整體統計：平均情感分數：{stats['avg_score']:.2f}, 正面：{stats['positive_count']}, 負面：{stats['negative_count']}, 中立：{stats['neutral_count']}
請提供：1. 整體聲譽趨勢判斷 2. 主要討論議題 3. 潛在風險或機會點"""
    
    @staticmethod
    def _parse_json_response(text: str) -> dict:
        try:
            clean_text = text.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return {}
    
    @staticmethod
    def _calculate_statistics(df: pd.DataFrame) -> dict:
        return {
            'avg_score': df['sentiment_score'].mean(),
            'positive_count': len(df[df['sentiment_label'] == '正面']),
            'negative_count': len(df[df['sentiment_label'] == '負面']),
            'neutral_count': len(df[df['sentiment_label'] == '中立'])
        }


# ============================================================================
# 第四部分：資料抓取服務 (支援地區篩選 + 程式碼優化)
# ============================================================================

class DataFetcher:
    """資料抓取基礎類別"""
    
    def __init__(self, analyzer: UnifiedAIAnalyzer):
        self.analyzer = analyzer
    
    def fetch(self, query: str, limit: int, region: str = "TW") -> list:
        raise NotImplementedError("子類別必須實作此方法")
    
    @staticmethod
    def _is_relevant(content: str, query: str, min_score: float = 0.3) -> bool:
        content_lower = content.lower()
        query_lower = query.lower()
        if query_lower in content_lower: return True
        
        keywords = query.replace('股份有限公司', '').replace('公司', '').strip()
        words = [w for w in keywords.split() if len(w) >= 2]
        if not words: words = [keywords]
        
        matched = sum(1 for word in words if word.lower() in content_lower)
        score = matched / len(words) if words else 0
        return score >= min_score
    
    def _create_result_dict(self, content: str, platform: str, url: str, 
                           author: str = "Unknown", engagement: int = 0) -> dict:
        score, label, topic = self.analyzer.analyze_sentiment(content)
        return {
            'content': content, 'platform': platform, 'score': score,
            'label': label, 'url': url, 'author': author, 'engagement': engagement
        }
    
    @staticmethod
    def _add_delay(min_sec: float = 1.0, max_sec: float = 2.0):
        time.sleep(random.uniform(min_sec, max_sec))


class RedditFetcher(DataFetcher):
    """Reddit 資料抓取器"""
    
    def __init__(self, analyzer, client_id, client_secret, user_agent):
        super().__init__(analyzer)
        import praw  # type: ignore
        self.reddit = praw.Reddit(
            client_id=client_id, client_secret=client_secret, user_agent=user_agent
        )
    
    def fetch(self, query: str, limit: int = 10, region: str = "TW") -> list:
        # Reddit 搜尋不強制區分 region (API 限制)，維持一般搜尋
        results = []
        try:
            for submission in self.reddit.subreddit("all").search(query, limit=limit, sort="new"):
                content = f"{submission.title}\n{submission.selftext[:200]}"
                result = self._create_result_dict(
                    content=content, platform='Reddit',
                    url=f"https://reddit.com{submission.permalink}",
                    author=str(submission.author),
                    engagement=submission.score + submission.num_comments
                )
                results.append(result)
                self._add_delay(1, 2)
        except Exception as e:
            st.error(f"Reddit 抓取失敗: {e}")
        return results


class YouTubeFetcher(DataFetcher):
    """YouTube 資料抓取器"""
    
    def __init__(self, analyzer, api_key):
        super().__init__(analyzer)
        from googleapiclient.discovery import build  # type: ignore
        self.youtube = build('youtube', 'v3', developerKey=api_key)
    
    def fetch(self, query: str, limit: int = 5, region: str = "TW") -> list:
        results = []
        try:
            region_code = 'TW' if region == 'TW' else None 
            
            search_response = self.youtube.search().list(
                q=query, part='id', maxResults=limit, type='video', 
                order='date', regionCode=region_code
            ).execute()
            
            for item in search_response.get('items', []):
                video_id = item['id']['videoId']
                results.extend(self._fetch_video_comments(video_id))
                self._add_delay(1, 1)
        except Exception as e:
            st.error(f"YouTube 抓取失敗: {e}")
        return results
    
    def _fetch_video_comments(self, video_id: str, max_comments: int = 3) -> list:
        results = []
        try:
            comments = self.youtube.commentThreads().list(
                part='snippet', videoId=video_id, maxResults=max_comments, order='relevance'
            ).execute()
            
            for comment_item in comments.get('items', []):
                comment = comment_item['snippet']['topLevelComment']['snippet']
                result = self._create_result_dict(
                    content=comment['textDisplay'][:300], platform='YouTube',
                    url=f"https://youtube.com/watch?v={video_id}",
                    author=comment['authorDisplayName'], engagement=comment['likeCount']
                )
                results.append(result)
        except: pass
        return results


class GoogleNewsFetcher(DataFetcher):
    """Google News RSS 抓取器 (支援地區) - 優化版"""
    
    def fetch(self, query: str, limit: int = 10, region: str = "TW") -> list:
        results = []
        try:
            # 設定語系參數
            lang_params = "hl=zh-TW&gl=TW&ceid=TW:zh-Hant" if region == "TW" else "hl=en-US&gl=US&ceid=US:en"
            
            # 內部函數：處理 URL 組合與抓取
            def _get_feed(search_query):
                encoded_query = quote(search_query)
                url = f"https://news.google.com/rss/search?q={encoded_query}&{lang_params}"
                return feedparser.parse(url)

            # 優先嘗試精確搜尋
            feed = _get_feed(f'"{query}"')
            
            # 若無結果，嘗試模糊搜尋
            if not feed.entries:
                st.warning(f"Google News ({region}) 未找到精確結果，嘗試模糊搜尋...")
                feed = _get_feed(query)
            
            relevant_count = 0
            for entry in feed.entries:
                if relevant_count >= limit: break
                
                title = entry.get('title', '')
                summary = entry.get('summary', entry.get('description', ''))
                content = f"{title}\n{summary[:200]}" if summary else title
                
                if not content.strip() or not self._is_relevant(content, query, 0.2): continue
                
                results.append(self._create_result_dict(
                    content=content, platform='Google News',
                    url=entry.get('link', entry.get('url', '')),
                    author=entry.get('source', {}).get('title', 'Unknown')
                ))
                relevant_count += 1
                self._add_delay(0.5, 1)
                
            if not results: st.info(f"🔍 Google News ({region}) 未找到相關新聞")
            
        except Exception as e:
            st.error(f"Google News 抓取失敗: {str(e)}")
            
        return results


class DuckDuckGoFetcher(DataFetcher):
    """DuckDuckGo 搜尋抓取器 (支援地區) - 優化版"""
    
    def __init__(self, analyzer):
        super().__init__(analyzer)
        self.last_request_time = 0
        self.min_interval = 3.5  # 增加間隔避免被封鎖
    
    def _rate_limit_wait(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
    
    def fetch(self, query: str, limit: int = 10, region: str = "TW", search_type: str = "news") -> list:
        results = []
        max_retries = 3
        ddg_region = 'tw-tz' if region == 'TW' else 'wt-wt'
        
        try:
            self._rate_limit_wait()
            with DDGS() as ddgs:
                search_results = []
                # 定義搜尋策略：[精確搜尋, 模糊搜尋]
                search_strategies = [f'"{query}"', query]
                
                for attempt in range(max_retries):
                    try:
                        if attempt > 0: st.info(f"🔄 DDG 重試中... ({attempt+1})")
                        
                        # 迴圈嘗試不同策略，減少 if/else 巢狀
                        for q_str in search_strategies:
                            if search_type == "news":
                                res = list(ddgs.news(
                                    query=q_str, region=ddg_region, safesearch='moderate', 
                                    timelimit='y', max_results=min(limit + 5, 15)  # 減少請求量
                                ))
                            else:
                                res = list(ddgs.text(
                                    query=q_str, region=ddg_region, safesearch='moderate', 
                                    backend='html', max_results=min(limit + 5, 15)  # 減少請求量
                                ))
                            
                            if res:
                                search_results = res
                                break # 成功獲得結果
                        
                        if search_results: break # 跳出重試迴圈
                        
                        time.sleep(3.0 + random.uniform(0, 1.5))  # 隨機延遲避免規律性
                        
                    except Exception as e:
                        st.warning(f"⚠️ DDG 嘗試 {attempt+1} 失敗: {str(e)[:80]}")
                        # 最後一次嘗試若是 News 失敗，降級為 Text (Fallback)
                        if attempt == max_retries - 1 and search_type == "news":
                            try:
                                search_results = list(ddgs.text(
                                    query=query, region=ddg_region, backend='html', max_results=min(limit + 5, 15)
                                ))
                            except Exception as fallback_err:
                                st.error(f"❌ Fallback 也失敗: {str(fallback_err)[:50]}")
                        time.sleep(3.0 + random.uniform(0, 1.5))

                # 處理結果
                relevant_count = 0
                for item in search_results:
                    if relevant_count >= limit: break
                    
                    title = item.get('title', '')
                    body = item.get('body', '')
                    content = f"{title}\n{body}"[:500]
                    url = item.get('url', item.get('href', ''))
                    
                    if not content.strip() or not url: continue
                    if not self._is_relevant(content, query, 0.1):  # 降低門檻從 0.2 到 0.1
                        continue
                    
                    results.append(self._create_result_dict(
                        content=content, platform='DuckDuckGo',
                        url=url, author=item.get('source', 'Web')
                    ))
                    relevant_count += 1
            
            # 提供搜尋結果統計
            if search_results and not results:
                st.warning(f"⚠️ DDG 回傳 {len(search_results)} 筆，但無符合相關性的結果")
            elif not search_results:
                st.warning("⚠️ DDG 未回傳任何結果，可能被限流或查詢過於頻繁")
                    
        except Exception as e:
            st.error(f"❌ DDG 連線錯誤: {str(e)[:120]}")
        
        return results


# ============================================================================
# 第五部分：資料抓取協調器
# ============================================================================

class DataCoordinator:
    """協調多個資料源的抓取作業"""
    
    def __init__(self, analyzer: UnifiedAIAnalyzer, config: Config):
        self.analyzer = analyzer
        self.config = config
        self.fetchers = self._initialize_fetchers()
    
    def _initialize_fetchers(self) -> dict:
        fetchers = {}
        fetchers['Google News'] = GoogleNewsFetcher(self.analyzer)
        fetchers['DuckDuckGo'] = DuckDuckGoFetcher(self.analyzer)
        
        if Config.REDDIT_AVAILABLE and self.config.REDDIT_CLIENT_ID:
            try:
                fetchers['Reddit'] = RedditFetcher(
                    self.analyzer, self.config.REDDIT_CLIENT_ID,
                    self.config.REDDIT_CLIENT_SECRET, self.config.REDDIT_USER_AGENT
                )
            except: pass
        
        if Config.YOUTUBE_AVAILABLE and self.config.YOUTUBE_API_KEY:
            try:
                fetchers['YouTube'] = YouTubeFetcher(self.analyzer, self.config.YOUTUBE_API_KEY)
            except: pass
        
        return fetchers
    
    def fetch_all(self, query: str, sources: list, items_per_source: int, region: str) -> list:
        all_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, source in enumerate(sources):
            status_text.text(f"正在抓取 {source} (地區: {region})...")
            
            if source in self.fetchers:
                results = self.fetchers[source].fetch(query, items_per_source, region=region)
                all_results.extend(results)
            
            progress_bar.progress((idx + 1) / len(sources))
        
        status_text.empty()
        progress_bar.empty()
        return all_results


# ============================================================================
# 第六部分：UI 介面
# ============================================================================

class StreamlitUI:
    """Streamlit 使用者介面"""
    
    def __init__(self):
        st.set_page_config(page_title="企業、品牌及個人聲譽智慧監測 V1.1", layout="wide", page_icon="🛡️")
        self.db = DatabaseManager()
        self.config = self._setup_sidebar()
        
        # 初始化 AI 分析器
        self.analyzer = None
        self.coordinator = None
        
        provider = st.session_state.get('llm_provider', 'Gemini')
        api_key = self.config.GEMINI_API_KEY if provider == 'Gemini' else self.config.OPENAI_API_KEY
        
        if api_key:
            self.analyzer = UnifiedAIAnalyzer(provider.lower(), api_key)
            self.coordinator = DataCoordinator(self.analyzer, self.config)
    
    def _setup_sidebar(self) -> Config:
        with st.sidebar:
            st.header("⚙️ 系統設定")
            
            # --- LLM 選擇 ---
            st.subheader("1. AI 模型選擇")
            llm_provider = st.radio(
                "選擇分析模型", 
                ["Gemini", "OpenAI"],
                key="llm_provider",
                help="Gemini 適合大量免費分析；OpenAI 需付費但穩定性高"
            )
            
            if llm_provider == "Gemini":
                Config.GEMINI_API_KEY = st.text_input("Gemini API Key", type="password")
            else:
                if not Config.OPENAI_AVAILABLE:
                    st.error("請先安裝：`pip install openai`")
                Config.OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
                
            st.divider()
            
            # --- 資料源 Key ---
            st.subheader("2. 資料源授權")
            with st.expander("Reddit / YouTube 設定"):
                Config.REDDIT_CLIENT_ID = st.text_input("Reddit Client ID", type="password")
                Config.REDDIT_CLIENT_SECRET = st.text_input("Reddit Client Secret", type="password")
                Config.YOUTUBE_API_KEY = st.text_input("YouTube API Key", type="password")
            
            st.divider()
            if st.button("🗑️ 清空資料庫", type="secondary"):
                self.db.clear_all_data()
                st.success("已清空")
                st.rerun()
        
        return Config
    
    def render_search_form(self) -> tuple:
        st.info("💡 提示：選擇「台灣」將鎖定繁體中文與本地新聞；「國際」將搜尋全球英文/外文資訊。")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            query = st.text_input("🔍 監控關鍵字", value="台北捷運", placeholder="輸入公司名稱")
        with col2:
            region = st.radio("🌍 搜尋區域", ["台灣 (TW)", "國際 (Global)"], horizontal=True)
            region_code = "TW" if "台灣" in region else "Global"

        col3, col4 = st.columns([2, 1])
        with col3:
            available_sources = ["Google News", "DuckDuckGo"]
            if Config.REDDIT_AVAILABLE: available_sources.insert(0, "Reddit")
            if Config.YOUTUBE_AVAILABLE: available_sources.insert(1, "YouTube")
            data_sources = st.multiselect("資料來源", available_sources, default=["Google News", "DuckDuckGo"])
        with col4:
            items_per_source = st.number_input("每來源筆數", 3, 20, 5)
            
        return query, data_sources, items_per_source, region_code
    
    def render_dashboard(self, df: pd.DataFrame):
        if df.empty:
            st.info("👆 請點擊上方按鈕開始監測")
            return
        
        # 報告區塊
        st.header(f"📊 AI 智慧分析報告 ({st.session_state.get('llm_provider', 'Unknown')})")
        with st.spinner("生成中..."):
            if self.analyzer:
                summary = self.analyzer.generate_summary_report(df)
                st.markdown(summary)
            else:
                st.warning("⚠️ 請先設定 API Key")
        
        # 關鍵指標
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("平均分數", f"{df['sentiment_score'].mean():.2f}")
        with col2: st.metric("正面", len(df[df['sentiment_label'] == '正面']))
        with col3: st.metric("負面", len(df[df['sentiment_label'] == '負面']))
        with col4: st.metric("互動數", f"{df['engagement'].sum():,}")
        
        # 圖表
        c1, c2 = st.columns(2)
        with c1: st.bar_chart(df['platform'].value_counts())
        with c2: st.bar_chart(df['sentiment_label'].value_counts())
        
        # 資料表
        st.dataframe(df[['timestamp', 'platform', 'sentiment_label', 'content', 'url']], 
                     column_config={"url": st.column_config.LinkColumn("連結")}, use_container_width=True)

    def run(self):
        st.title("🛡️ 企業、品牌及個人聲譽智慧監測 V1.1")
        st.caption("版本：Gemini 支援 | OpenAI 支援 | 地區切換 | Google News | DuckDuckGo | Reddit | YouTube")
        
        query, data_sources, items_per_source, region = self.render_search_form()
        
        if st.button("🚀 開始智慧監測", type="primary"):
            provider = st.session_state.get('llm_provider')
            has_key = Config.GEMINI_API_KEY if provider == 'Gemini' else Config.OPENAI_API_KEY
            
            if not has_key:
                st.error(f"⚠️ 請先在側邊欄輸入 {provider} API Key")
            else:
                with st.spinner(f"🤖 {provider} AI 正在搜尋 ({region}) 並分析..."):
                    all_results = self.coordinator.fetch_all(
                        query, data_sources, items_per_source, region
                    )
                    for item in all_results:
                        self.db.save_mention(
                            item['content'], item['platform'], item['score'],
                            item['label'], item['url'], item['author'], item['engagement']
                        )
                    st.success(f"✅ 完成！共分析 {len(all_results)} 筆資料")
                    st.rerun()
        
        df = self.db.get_recent_mentions(limit=100)
        self.render_dashboard(df)

if __name__ == "__main__":
    app = StreamlitUI()
    app.run()