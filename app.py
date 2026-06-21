from flask import Flask, render_template, request, jsonify
import sys, os, warnings
import numpy as np
import requests as req_lib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

from main import analyze_market, analyze_technicals, check_canslim, score_and_risk, analyze_whale, _rsi, _macd
import yfinance as yf

app = Flask(__name__)

KR_NAME_MAP = {
    # 반도체·AI
    "엔비디아": "NVDA", "인텔": "INTC", "amd": "AMD", "퀄컴": "QCOM",
    "브로드컴": "AVGO", "마이크론": "MU", "텍사스인스트루먼트": "TXN",
    "어플라이드머티리얼즈": "AMAT", "램리서치": "LRCX", "kla": "KLAC",
    "아이온큐": "IONQ", "퀀텀컴퓨팅": "QUBT", "리게티": "RGTI", "디웨이브": "QBTS",
    "허니웰": "HON",
    # AI·클라우드·SaaS
    "마이크로소프트": "MSFT", "구글": "GOOGL", "알파벳": "GOOGL",
    "아마존": "AMZN", "메타": "META", "오라클": "ORCL",
    "세일즈포스": "CRM", "어도비": "ADBE", "팔로알토": "PANW",
    "서비스나우": "NOW", "워크데이": "WDAY", "스노우플레이크": "SNOW",
    "몽고db": "MDB", "데이터도그": "DDOG", "클라우드플레어": "NET",
    "허브스팟": "HUBS", "옥타": "OKTA", "c3ai": "AI", "빅베어ai": "BBAI",
    "사운드하운드": "SOUN", "앱러빈": "APP", "업스타트": "UPST",
    "줌인포": "ZI", "트윌리오": "TWLO", "도큐사인": "DOCU",
    # 빅테크
    "애플": "AAPL", "넷플릭스": "NFLX",
    # EV·항공·우주
    "테슬라": "TSLA", "리비안": "RIVN", "루시드": "LCID",
    "니오": "NIO", "샤오펑": "XPEV", "리오토": "LI",
    "퀀텀스케이프": "QS", "솔리드파워": "SLDP",
    "조비에이비에션": "JOBY", "조비": "JOBY",
    "아처에이비에션": "ACHR", "아처": "ACHR",
    "버티칼에어로스페이스": "EVTL", "블레이드": "BLDE",
    "릴리움": "LILM", "위스크에어로": "JOBY",
    "로켓랩": "RKLB", "버진갤럭틱": "SPCE", "아스트라": "ASTR",
    "레드와이어": "RDW", "맥사": "MAXR",
    # 모빌리티
    "우버": "UBER", "리프트": "LYFT",
    # 핀테크·크립토
    "코인베이스": "COIN", "팔란티어": "PLTR", "블록": "SQ", "스퀘어": "SQ",
    "소파이": "SOFI", "로빈후드": "HOOD", "아펌": "AFRM", "클라나": "KLAR",
    "마이크로스트래티지": "MSTR",
    # 바이오·헬스
    "화이자": "PFE", "모더나": "MRNA", "바이오엔텍": "BNTX", "길리어드": "GILD",
    "암젠": "AMGN", "존슨앤존슨": "JNJ", "존슨앤드존슨": "JNJ",
    "일라이릴리": "LLY", "노보노디스크": "NVO", "아스트라제네카": "AZN",
    "버텍스": "VRTX", "리제네론": "REGN", "바이오젠": "BIIB",
    # 소비·엔터·여행
    "디즈니": "DIS", "스타벅스": "SBUX", "나이키": "NKE", "맥도날드": "MCD",
    "코카콜라": "KO", "펩시": "PEP", "월마트": "WMT", "타겟": "TGT",
    "홈디포": "HD", "에어비앤비": "ABNB", "부킹홀딩스": "BKNG",
    "익스피디아": "EXPE", "델타항공": "DAL", "유나이티드항공": "UAL",
    "아메리칸항공": "AAL", "사우스웨스트항공": "LUV",
    # 에너지·산업
    "엑슨모빌": "XOM", "쉐브론": "CVX", "보잉": "BA", "록히드마틴": "LMT",
    "레이시온": "RTX", "캐터필러": "CAT", "디어": "DE",
    "퍼스트솔라": "FSLR", "선파워": "SPWR", "플러그파워": "PLUG",
    "블룸에너지": "BE", "퓨얼셀에너지": "FCEL",
    # 금융
    "jp모건": "JPM", "골드만삭스": "GS", "모건스탠리": "MS",
    "뱅크오브아메리카": "BAC", "씨티그룹": "C", "웰스파고": "WFC",
    "버크셔해서웨이": "BRK-B", "버크셔": "BRK-B",
    "블랙록": "BLK", "찰스슈왑": "SCHW",
    # 통신·미디어
    "버라이즌": "VZ", "at&t": "T",
    # SNS·커머스
    "스냅": "SNAP", "핀터레스트": "PINS", "스포티파이": "SPOT",
    "줌": "ZM", "쇼피파이": "SHOP", "이베이": "EBAY",
    "에치": "ETSY", "엣시": "ETSY", "도어대시": "DASH", "인스타카트": "CART",
    # 게임
    "로블록스": "RBLX", "유니티": "U", "ea": "EA", "액티비전": "ATVI",
    "테이크투": "TTWO",
    # 방산·우주 (추가)
    "노스롭그루먼": "NOC", "제너럴다이나믹스": "GD", "l3해리스": "LHX",
    # 한국 주요 종목
    "삼성전자": "005930.KS", "sk하이닉스": "000660.KS", "lg에너지솔루션": "373220.KS",
    "현대차": "005380.KS", "현대자동차": "005380.KS", "기아": "000270.KS",
    "셀트리온": "068270.KS", "카카오": "035720.KS", "네이버": "035420.KS",
    "lg화학": "051910.KS", "삼성바이오로직스": "207940.KS", "포스코": "005490.KS",
    "현대모비스": "012330.KS", "삼성sdi": "006400.KS", "kb금융": "105560.KS",
    "신한지주": "055550.KS", "하나금융지주": "086790.KS", "카카오뱅크": "323410.KS",
    "크래프톤": "259960.KS", "엔씨소프트": "036570.KS", "넷마블": "251270.KS",
    "두산에너빌리티": "034020.KS", "한화에어로스페이스": "012450.KS",
    "lg전자": "066570.KS", "sk이노베이션": "096770.KS", "롯데케미칼": "011170.KS",
}


def to_json(obj):
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_insight(name, tech, risk):
    sc = risk["score"]
    s200 = tech.get("s200")
    below_warn = (
        f" 단, 200일선({s200:,.2f}) 아래에 위치해 있어 추가 하락 리스크가 있다. "
        f"포지션은 절반으로 줄이고, 손절을 평소보다 타이트하게 유지하라."
        if not tech["above_200"] and s200 else ""
    )
    if sc >= 90:
        return (f"'{name}'은(는) 퀀트 스코어 {sc}점의 최고 등급 셋업이다. "
                f"시장이 지지하고 거래량이 확인된다면 주저 없이 진입하라. "
                f"이런 기회는 1년에 몇 번 오지 않는다." + below_warn)
    if sc >= 80:
        return (f"'{name}'의 {sc}점 셋업은 충분히 매력적이다. 하프 포지션으로 "
                f"진입하고, 1차 목표가 도달 시 손절가를 본전으로 올려 리스크 제로를 만들어라." + below_warn)
    if sc >= 70:
        return (f"'{name}'은(는) 가능성은 있으나 확신이 부족한 {sc}점 구간이다. "
                f"소규모 테스트 포지션으로 시장 반응을 확인하고 전략을 재평가하라." + below_warn)
    if sc >= 50:
        return (f"'{name}'의 {sc}점은 진입 기준 미달이다. 현재 손익비가 불리하다 — "
                f"인내심을 갖고 더 나은 타점을 기다려라. 기다림도 전략이다.")
    return (f"'{name}'의 {sc}점 신호는 명백한 청산 신호다. 보유 중이라면 "
            f"손실이 커지기 전에 출구를 찾아라. 손절은 패배가 아닌 자본 보호다.")


PRIVATE_COMPANIES = {
    "스페이스x": "SpaceX", "스페이스엑스": "SpaceX",
    "오픈ai": "OpenAI", "챗gpt": "OpenAI",
    "앤트로픽": "Anthropic",
    "데이터브릭스": "Databricks",
    "스트라이프": "Stripe",
    "틱톡": "TikTok / ByteDance",
    "바이트댄스": "ByteDance",
    "샤인": "SHEIN",
    "레볼루트": "Revolut",
    "클라나": "Klarna",
    "패스트": "Fast",
}

@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    q_lower = q.lower().replace(" ", "")

    # 비상장 기업 체크
    private_match = next((v for k, v in PRIVATE_COMPANIES.items() if k.startswith(q_lower)), None)
    if private_match:
        return jsonify([{
            "symbol": "__PRIVATE__",
            "name": f"{private_match} — 비상장 기업으로 주식 분석이 불가합니다",
            "exchange": "", "type": "",
        }])

    # 한국어 종목명 딕셔너리 우선 조회
    matched = [(k, v) for k, v in KR_NAME_MAP.items() if k.replace(" ", "").startswith(q_lower)]
    if matched:
        results = []
        for kr_name, symbol in matched[:8]:
            try:
                info = yf.Ticker(symbol).info or {}
                eng_name = info.get("longName") or info.get("shortName") or kr_name
            except Exception:
                eng_name = kr_name
            results.append({
                "symbol":   symbol,
                "name":     f"{kr_name} ({eng_name})",
                "exchange": "KRX" if symbol.endswith(".KS") else "US",
                "type":     "Equity",
            })
        return jsonify(results)

    is_korean = any('가' <= c <= '힣' for c in q)
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {"q": q, "quotesCount": 8, "newsCount": 0, "enableFuzzyQuery": "false"}
        headers = {"User-Agent": "Mozilla/5.0"}
        r = req_lib.get(url, params=params, headers=headers, timeout=5)
        quotes = r.json().get("quotes", [])
        results = []
        for item in quotes:
            if item.get("typeDisp") in ("Equity", "ETF"):
                results.append({
                    "symbol":   item.get("symbol", ""),
                    "name":     item.get("longname") or item.get("shortname") or "",
                    "exchange": item.get("exchDisp", ""),
                    "type":     item.get("typeDisp", ""),
                })
        if not results and is_korean:
            results.append({
                "symbol": "__HINT__",
                "name": f"'{q}' 검색 결과 없음 — 영문명이나 티커로 검색해보세요",
                "exchange": "", "type": "",
            })
        return jsonify(results)
    except Exception:
        return jsonify([])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        ticker = (data.get("ticker") or "").strip().upper()
        capital = float(data.get("capital") or 10_000_000)

        if not ticker:
            return jsonify({"error": "티커를 입력해주세요."}), 400

        stock = yf.Ticker(ticker)
        df = stock.history(period="2y")

        if df.empty:
            return jsonify({"error": f"'{ticker}' 데이터를 가져올 수 없습니다. 티커를 확인해주세요."}), 404

        df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index

        info = {}
        try:
            info = stock.info or {}
        except Exception:
            pass

        market = analyze_market()
        tech = analyze_technicals(df)
        canslim, canslim_score = check_canslim(info, df, market["market_trend"])
        risk = score_and_risk(tech, canslim_score, capital)
        whale = analyze_whale(df, info)

        name = info.get("longName") or info.get("shortName") or ticker
        insight = get_insight(name, tech, risk)

        risk["detail"] = {k: list(v) for k, v in risk["detail"].items()}

        # Chart data (1년치 OHLCV + MA)
        chart_df = df[df['Volume'] > 0].tail(365)

        def date_str(ts):
            return ts.strftime('%Y-%m-%d')

        ohlcv = [
            {"time": date_str(ts),
             "open":  round(float(row['Open']),  4),
             "high":  round(float(row['High']),  4),
             "low":   round(float(row['Low']),   4),
             "close": round(float(row['Close']), 4),
             "volume": int(row['Volume'])}
            for ts, row in chart_df.iterrows()
        ]

        def ma_line(window):
            ma = df['Close'].rolling(window).mean()
            return [
                {"time": date_str(ts), "value": round(float(v), 4)}
                for ts, v in ma.items()
                if ts in chart_df.index and not np.isnan(v)
            ]

        def indicator_series(series):
            return [
                {"time": date_str(ts), "value": round(float(v), 4)}
                for ts, v in series.items()
                if ts in chart_df.index and not np.isnan(v)
            ]

        close_s = df['Close'].astype(float)
        rsi_s = _rsi(close_s, 14)
        macd_line_s, macd_sig_s, macd_hist_s = _macd(close_s)

        result = {
            "ticker": ticker,
            "name": name,
            "sector": info.get("sector") or info.get("industry") or "N/A",
            "currency": info.get("currency") or "USD",
            "market": market,
            "tech": tech,
            "canslim": canslim,
            "canslim_score": int(canslim_score),
            "risk": risk,
            "whale": whale,
            "insight": insight,
            "chart": {
                "ohlcv":       ohlcv,
                "sma10":       ma_line(10),
                "sma20":       ma_line(20),
                "sma50":       ma_line(50),
                "sma200":      ma_line(200),
                "rsi":         indicator_series(rsi_s),
                "macd_line":   indicator_series(macd_line_s),
                "macd_signal": indicator_series(macd_sig_s),
                "macd_hist": [
                    {"time": date_str(ts), "value": round(float(v), 4),
                     "color": "rgba(63,185,80,0.7)" if v >= 0 else "rgba(248,81,73,0.7)"}
                    for ts, v in macd_hist_s.items()
                    if ts in chart_df.index and not np.isnan(v)
                ],
            },
        }

        return jsonify(to_json(result))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


SCAN_TICKERS = {
    "us": [
        # 빅테크
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","BRK-B","V","JPM",
        # 반도체·AI
        "AMD","INTC","QCOM","AVGO","MU","AMAT","ARM","SMCI","TSM",
        # AI 소프트·클라우드
        "PLTR","AI","SOUN","BBAI","APP","SNOW","NET","DDOG","CRM","ORCL",
        # 퀀텀컴퓨팅
        "IONQ","RGTI","QBTS","QUBT",
        # EV·항공우주
        "RIVN","LCID","NIO","XPEV","JOBY","ACHR","RKLB",
        # 바이오·헬스
        "LLY","NVO","MRNA","PFE","AMGN","GILD",
        # 핀테크·크립토
        "COIN","MSTR","HOOD","SOFI","AFRM",
        # 소비·엔터
        "NFLX","DIS","SBUX","NKE","ABNB",
        # ETF
        "SPY","QQQ","SOXL","ARKK",
    ],
    "kr": [
        "005930.KS","000660.KS","373220.KS","005380.KS","000270.KS",
        "068270.KS","035720.KS","035420.KS","051910.KS","207940.KS",
        "012450.KS","259960.KS","034020.KS","006400.KS","066570.KS",
        "105560.KS","055550.KS","323410.KS","036570.KS","012330.KS",
    ],
}

_market_cache = {"data": None}

@app.route("/scan", methods=["POST"])
def scan():
    data      = request.get_json() or {}
    market    = data.get("market", "all")
    min_score = int(data.get("min_score", 0))
    sig_filter = data.get("signal", "all")   # all / buy / hold / sell
    above_filter = data.get("above_200", "all")  # all / yes / no
    capital   = float(data.get("capital", 10_000_000))

    if market == "us":
        tickers = SCAN_TICKERS["us"]
    elif market == "kr":
        tickers = SCAN_TICKERS["kr"]
    else:
        tickers = SCAN_TICKERS["us"] + SCAN_TICKERS["kr"]

    # 시장 데이터는 1번만 호출
    try:
        from main import analyze_market as _am
        mkt = _am()
    except Exception:
        mkt = {"market_trend": "N/A", "vix": 0, "vix_status": "N/A", "dist_days": 0}

    def analyze_one(ticker):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            if df.empty or len(df) < 60:
                return None
            df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
            info = {}
            try:
                info = stock.info or {}
            except Exception:
                pass
            tech = analyze_technicals(df)
            _, cs = check_canslim(info, df, mkt["market_trend"])
            risk = score_and_risk(tech, cs, capital)
            return {
                "ticker":    ticker,
                "name":      info.get("longName") or info.get("shortName") or ticker,
                "sector":    info.get("sector") or info.get("industry") or "N/A",
                "currency":  info.get("currency") or ("KRW" if ".KS" in ticker else "USD"),
                "cur":       round(tech["cur"], 2),
                "score":     risk["score"],
                "signal":    risk["signal"],
                "action":    risk["action"],
                "rsi":       tech["rsi"],
                "above_200": tech["above_200"],
                "macd_st":   tech["macd_st"],
                "vol_ratio": tech["vol_ratio"],
                "stage":     tech["stage"],
            }
        except Exception:
            return None

    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(analyze_one, t): t for t in tickers}
        for f in as_completed(futures):
            try:
                r = f.result(timeout=25)
                if r:
                    results.append(r)
            except Exception:
                pass

    # 필터 적용
    if min_score > 0:
        results = [r for r in results if r["score"] >= min_score]
    if sig_filter == "buy":
        results = [r for r in results if "매수" in r["signal"]]
    elif sig_filter == "hold":
        results = [r for r in results if "관망" in r["signal"]]
    elif sig_filter == "sell":
        results = [r for r in results if "매도" in r["signal"]]
    if above_filter == "yes":
        results = [r for r in results if r["above_200"]]
    elif above_filter == "no":
        results = [r for r in results if not r["above_200"]]

    results.sort(key=lambda x: x["score"], reverse=True)
    return jsonify(to_json(results))


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
