#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 📊  주식 심층 분석 프로그램
     CANSLIM + 퀀트 알파 프로토콜 (6단계)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import io
import argparse
import warnings
from datetime import datetime

# Windows 콘솔 UTF-8 출력 강제
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


def _check_libs():
    missing = []
    for pkg in ["yfinance", "pandas", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[오류] 필요한 라이브러리가 없습니다: {', '.join(missing)}")
        print(f"       설치: pip install {' '.join(missing)}")
        sys.exit(1)

_check_libs()

import yfinance as yf  # noqa: E402
import pandas as pd    # noqa: E402
import numpy as np     # noqa: E402


# ──────────────────────────────────────────────────
# 기술적 지표 계산
# ──────────────────────────────────────────────────

def _sma(s, n):
    return s.rolling(n, min_periods=1).mean()

def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def _rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n, min_periods=1).mean()
    l = (-d.clip(upper=0)).rolling(n, min_periods=1).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _obv(close, vol):
    return (np.sign(close.diff().fillna(0)) * vol).cumsum()

def _bb(s, n=20, k=2):
    mid = _sma(s, n)
    std = s.rolling(n, min_periods=1).std()
    upper = mid + k * std
    lower = mid - k * std
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    bw = (upper - lower) / mid.replace(0, np.nan)
    return mid, upper, lower, pct_b, bw

def _macd(s, f=12, sl=26, sig=9):
    ml = _ema(s, f) - _ema(s, sl)
    sl_ = _ema(ml, sig)
    return ml, sl_, ml - sl_


# ──────────────────────────────────────────────────
# 0단계: 시장 환경 (Market Pulse)
# ──────────────────────────────────────────────────

def analyze_market():
    result = {"vix": 0.0, "vix_status": "N/A", "market_trend": "N/A", "dist_days": 0}
    try:
        vix = yf.Ticker("^VIX").history(period="5d")["Close"]
        v = float(vix.iloc[-1])
        result["vix"] = round(v, 2)
        result["vix_status"] = (
            "🔴 위험 (포지션 50% 강제 축소)" if v >= 30
            else "🟡 경계" if v >= 20
            else "🟢 정상"
        )
    except Exception:
        pass

    try:
        sp = yf.Ticker("^GSPC").history(period="6mo")["Close"]
        cur = float(sp.iloc[-1])
        s50 = float(_sma(sp, 50).iloc[-1])
        s200 = float(_sma(sp, 200).iloc[-1]) if len(sp) >= 200 else None
        dist = int((sp.pct_change().iloc[-25:] < -0.002).sum())
        result["dist_days"] = dist
        if cur > s50 and (s200 is None or cur > s200):
            result["market_trend"] = "Confirmed Uptrend ✅"
        elif cur > s50:
            result["market_trend"] = "Under Pressure ⚠️"
        else:
            result["market_trend"] = "Correction ⛔"
    except Exception:
        pass

    return result


# ──────────────────────────────────────────────────
# 1~3단계: 기술적 분석
# ──────────────────────────────────────────────────

def analyze_technicals(df):
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    s10  = _sma(close, 10)
    s20  = _sma(close, 20)
    s50  = _sma(close, 50)
    s200 = _sma(close, 200)

    cur  = float(close.iloc[-1])
    v10  = float(s10.iloc[-1])
    v20  = float(s20.iloc[-1])
    v50  = float(s50.iloc[-1])
    v200 = float(s200.iloc[-1]) if len(close) >= 50 else None

    above_200 = cur > v200 if v200 else False

    # 이평선 정배열
    if v200 and v20 > v50 > v200:
        ma_align, align_g = "정배열 ✅ (20>50>200)", "bullish"
    elif v200 and v20 < v50 < v200:
        ma_align, align_g = "역배열 ⛔ (20<50<200)", "bearish"
    else:
        ma_align, align_g = "혼조세 ⚠️", "mixed"

    # Stage 분석 (스탠 와인스타인)
    rising_20d = cur > float(close.iloc[-21]) if len(close) > 21 else False
    if v200:
        if above_200 and cur > v50 and rising_20d:
            stage, stage_n = "Stage 2 📈 (상승 추세 — 매수 구간)", 2
        elif above_200 and cur > v50:
            stage, stage_n = "Stage 3 ⚠️ (천장권 횡보)", 3
        elif above_200:
            stage, stage_n = "Stage 1 ↔️ (횡보 — 베이스 형성)", 1
        else:
            stage, stage_n = "Stage 4 📉 (하락 추세)", 4
    else:
        stage, stage_n = "데이터 부족 (200일선 미형성)", 0

    # RSI
    rsi_s   = _rsi(close, 14)
    rsi_cur = float(rsi_s.iloc[-1])
    rsi_5   = float(rsi_s.iloc[-6]) if len(rsi_s) >= 6 else rsi_cur

    if 50 <= rsi_cur <= 70:
        rsi_status = "건강 구간 (50~70)"
    elif rsi_cur > 70:
        rsi_status = "강력 추세 / 과매수"
    elif rsi_cur < 30:
        rsi_status = "과매도"
    else:
        rsi_status = "약세 구간"

    p_dir = 1 if cur > float(close.iloc[-15]) else -1
    r_dir = 1 if rsi_cur > rsi_5 else -1
    if p_dir > 0 and r_dir < 0:
        divergence = "하락 다이버전스 ⚠️ (경계)"
    elif p_dir < 0 and r_dir > 0:
        divergence = "상승 다이버전스 ✅ (반전 예고)"
    else:
        divergence = "없음"

    # OBV
    obv_s    = _obv(close, vol)
    obv_cur  = float(obv_s.iloc[-1])
    obv_ma20 = float(_sma(obv_s, 20).iloc[-1])
    obv_trend = "상승 (강세)" if obv_cur > obv_ma20 else "하락 (약세)"

    p20 = (float(close.iloc[-1]) - float(close.iloc[-21])) / float(close.iloc[-21]) if len(close) > 21 else 0
    o20 = float(obv_s.iloc[-21]) if len(obv_s) > 21 else obv_cur
    obv_bull_div = p20 <= 0.02 and obv_cur > o20

    # 거래량
    vol_avg20 = float(vol.rolling(20, min_periods=1).mean().iloc[-1])
    vol_ratio = (float(vol.iloc[-1]) / vol_avg20 * 100) if vol_avg20 > 0 else 100.0

    # 볼린저 밴드
    _, _, _, pct_b_s, bw_s = _bb(close)
    bw_cur  = float(bw_s.iloc[-1]) if not pd.isna(bw_s.iloc[-1]) else 0
    bw_ma20 = float(_sma(bw_s.dropna(), 20).iloc[-1]) if len(bw_s.dropna()) >= 20 else bw_cur
    pct_b   = float(pct_b_s.iloc[-1]) if not pd.isna(pct_b_s.iloc[-1]) else 0.5
    bb_squeeze    = bw_cur < bw_ma20 * 0.8
    bb_expansion  = bw_cur > bw_ma20 * 1.2

    # MACD
    ml, sl_, _ = _macd(close)
    mc, sc = float(ml.iloc[-1]), float(sl_.iloc[-1])
    mp, sp = float(ml.iloc[-2]) if len(ml) >= 2 else mc, float(sl_.iloc[-2]) if len(sl_) >= 2 else sc
    if mc > sc and mp <= sp:
        macd_st = "골든크로스 ✅"
    elif mc < sc and mp >= sp:
        macd_st = "데드크로스 ⛔"
    elif mc > sc:
        macd_st = "골든크로스 유지 ✅"
    else:
        macd_st = "데드크로스 유지 ⛔"

    recent_low   = float(low.iloc[-20:].min())
    week52_high  = float(high.iloc[-252:].max()) if len(high) >= 252 else float(high.max())

    return dict(
        cur=cur, s10=v10, s20=v20, s50=v50, s200=v200,
        above_200=above_200, ma_align=ma_align, align_g=align_g,
        stage=stage, stage_n=stage_n,
        rsi=round(rsi_cur, 2), rsi_status=rsi_status, divergence=divergence,
        obv_trend=obv_trend, obv_bull_div=obv_bull_div,
        vol_ratio=round(vol_ratio, 1),
        bb_squeeze=bb_squeeze, bb_expansion=bb_expansion, pct_b=round(pct_b, 3),
        macd_st=macd_st,
        recent_low=recent_low, week52_high=week52_high,
    )


# ──────────────────────────────────────────────────
# CANSLIM 체크 (3단계 ⑤)
# ──────────────────────────────────────────────────

def check_canslim(info, df, market_trend):
    close = df["Close"].astype(float)
    c = {}

    eg  = info.get("earningsGrowth") or 0
    eps = info.get("trailingEps") or 0
    fwd = info.get("forwardEps") or 0
    c["C"] = eg >= 0.25 or (eps > 0 and fwd > eps * 1.25)

    c["A"] = eg > 0.10 or (info.get("revenueGrowth") or 0) > 0.10

    h52 = info.get("fiftyTwoWeekHigh") or float(close.max())
    c["N"] = float(close.iloc[-1]) >= h52 * 0.95

    inst = info.get("heldPercentInstitutions") or 0
    c["S"] = inst > 0.30

    ret1y = (float(close.iloc[-1]) - float(close.iloc[0])) / float(close.iloc[0]) if len(close) > 1 else 0
    c["L"] = ret1y > 0.20

    c["I"] = inst > 0.20

    c["M"] = "Confirmed" in market_trend or "Uptrend" in market_trend

    return c, sum(1 for v in c.values() if v)


# ──────────────────────────────────────────────────
# 5단계: 세력 추적 (Whale Tracker)
# ──────────────────────────────────────────────────

def analyze_whale(df, info):
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    vol_ma20 = vol.rolling(20, min_periods=1).mean()
    sigs = []

    # ① 매집/분산 압력 — 최근 20일 상승일 vs 하락일 거래량 비교
    diff = close.diff()
    up_vol = vol[diff > 0].iloc[-20:]
    dn_vol = vol[diff < 0].iloc[-20:]
    if len(up_vol) >= 3 and len(dn_vol) >= 3:
        uv = float(up_vol.mean())
        dv = float(dn_vol.mean())
        ratio = uv / dv if dv > 0 else 1
        if ratio >= 1.5:
            sigs.append(f"① 매집 압력 감지 🟡 (상승일 거래량이 하락일 대비 {ratio:.1f}배 — 세력 매집 징후)")
        elif ratio <= 0.65:
            sigs.append(f"① 분산 압력 감지 🔴 (하락일 거래량이 상승일 대비 {1/ratio:.1f}배 — 세력 출구 징후)")

    # ② 조용한 매집 — 최근 5일 거래량 1.5배+ 이면서 가격 변동 3% 미만
    r5_vol = float(vol.iloc[-5:].mean())
    r5_avg = float(vol_ma20.iloc[-5:].mean())
    p5_chg = abs((float(close.iloc[-1]) - float(close.iloc[-6])) / float(close.iloc[-6])) if len(close) >= 6 else 1
    if r5_vol > r5_avg * 1.5 and p5_chg < 0.03:
        sigs.append("② 조용한 매집 감지 🟡 (고거래량 + 소폭 가격 변동 — 횡보 중 매집)")

    # ③ 최근 60일 이상 거래량 스파이크 (3배+)
    recent_ratio = (vol / vol_ma20).iloc[-60:]
    spikes = int((recent_ratio > 3).sum())
    if spikes >= 1:
        max_ratio = float(recent_ratio.max())
        sigs.append(f"③ 이상 거래량 스파이크 {spikes}회 감지 🟡 (최대 {max_ratio:.1f}배 — 세력 대량 매매 흔적)")

    # ④ 고점 이후 분산 완료 패턴 (최근 40일)
    if len(close) >= 40:
        w = close.iloc[-40:]
        pk = int(w.values.argmax())
        if 5 <= pk <= 34:
            vb = float(vol.iloc[-40:-40+pk+1].mean())
            va = float(vol.iloc[-40+pk+1:].mean())
            price_drop = float(close.iloc[-1]) < float(w.iloc[pk]) * 0.95
            if vb > 0 and va < vb * 0.6 and price_drop:
                sigs.append("④ 분산 완료 패턴 🔴 (고점 이후 거래량 급감 + 가격 하락 — 세력 출구)")

    # ⑤ 포켓 피벗 — 오늘 상승 + 거래량이 최근 10일 하락일 최대치 초과 (기관 매수 신호)
    if len(vol) >= 12 and diff.iloc[-1] > 0:
        past_dn_vol = vol[diff < 0].iloc[-11:-1]
        if len(past_dn_vol) > 0:
            max_dn = float(past_dn_vol.max())
            today_v = float(vol.iloc[-1])
            if today_v > max_dn:
                sigs.append(f"⑤ 포켓 피벗 감지 ✅ (상승일 거래량 {today_v/max_dn:.1f}배 — 기관 적극 매수 신호)")

    # ⑥ 절정 거래량 (Climactic Run) — 최근 5일 최대 대비 2배+ 폭등/폭락
    if len(vol) >= 6:
        prev5_max = float(vol.iloc[-6:-1].max())
        today_v   = float(vol.iloc[-1])
        today_chg = abs(float(close.pct_change().iloc[-1]))
        if today_v > prev5_max * 2 and today_chg > 0.03:
            sigs.append("⑥ 절정 거래량 감지 🔴 (극단적 거래량 + 급등락 — 추세 전환 경계)")

    # ⑦ 장기 OBV 다이버전스 (90일)
    if len(close) >= 90:
        obv_s = _obv(close, vol)
        p90 = float(close.iloc[-1]) - float(close.iloc[-90])
        o90 = float(obv_s.iloc[-1]) - float(obv_s.iloc[-90])
        if p90 < 0 and o90 > 0:
            sigs.append("⑦ 강세 OBV 다이버전스 ✅ (90일간 가격 하락 중 OBV 상승 — 비밀 매집)")
        elif p90 > 0 and o90 < 0:
            sigs.append("⑦ 약세 OBV 다이버전스 ⚠️ (90일간 가격 상승 중 OBV 하락 — 세력 분산)")

    # ⑧ 작전주 3박자 (소형주 + 단기 급등)
    mc  = info.get("marketCap") or 0
    p1m = (float(close.iloc[-1]) - float(close.iloc[-22])) / float(close.iloc[-22]) * 100 if len(close) >= 22 else 0
    if 0 < mc < 500_000_000_000 and p1m > 30:
        sigs.append("⑧ 작전주 패턴 경계 🚨 (소형주 + 단기 30%+ 급등)")

    if not sigs:
        sigs.append("특이 세력 흔적 없음 🟢")
        return {"sigs": sigs, "intent": "관망", "alert": "🟢 정상"}

    if any("작전주" in s for s in sigs):
        intent, alert = "작전주 경계", "🔴 위험"
    elif any("분산 완료" in s or "절정 거래량" in s for s in sigs):
        intent, alert = "분산 중 (출구 경계)", "🔴 위험"
    elif any("분산 압력" in s or "약세 OBV" in s for s in sigs):
        intent, alert = "분산 징후", "🟡 주의"
    elif any("포켓 피벗" in s for s in sigs):
        intent, alert = "기관 적극 매수 중", "🟢 긍정적"
    elif any("매집" in s or "강세 OBV" in s for s in sigs):
        intent, alert = "저가 매집 중", "🟡 주의"
    elif any("이상 거래량" in s for s in sigs):
        intent, alert = "세력 개입 흔적 있음", "🟡 주의"
    else:
        intent, alert = "불명확", "🟡 주의"

    return {"sigs": sigs, "intent": intent, "alert": alert}


# ──────────────────────────────────────────────────
# 4단계 + 6단계: 퀀트 스코어 & 리스크 관리
# ──────────────────────────────────────────────────

def score_and_risk(tech, canslim_score, capital):
    score = 0
    detail = {}

    def add(label, pts, max_pts):
        nonlocal score
        score += pts
        detail[label] = (pts, max_pts)

    # ① 200일선 위 (20점)
    add("200일선 위", 20 if tech["above_200"] else 0, 20)

    # ② 이평선 정배열 (15점)
    g = tech["align_g"]
    add("이평선 정배열", 15 if g == "bullish" else 5 if g == "mixed" else 0, 15)

    # ③ 거래량 돌파 (15점)
    vr = tech["vol_ratio"]
    add("거래량 돌파", 15 if vr >= 150 else 8 if vr >= 100 else 0, 15)

    # ④ RSI 상태 (10점)
    r = tech["rsi"]
    add("RSI 상태", 10 if 50 <= r <= 70 else 8 if r > 70 else 2 if r < 30 else 0, 10)

    # ⑤ OBV 다이버전스 (15점)
    add("OBV 다이버전스",
        15 if tech["obv_bull_div"] else 8 if tech["obv_trend"] == "상승 (강세)" else 0, 15)

    # ⑥ 패턴 완성도 (10점)
    add("패턴 완성도",
        10 if tech["bb_expansion"] and tech["above_200"] else 5 if tech["bb_squeeze"] else 0, 10)

    # ⑦ CANSLIM 보너스 (15점)
    add("CANSLIM 보너스", 15 if canslim_score >= 5 else 0, 15)

    # 손절가 & 목표가
    cur  = tech["cur"]
    sl   = min(max(tech["recent_low"], cur * 0.92), cur * 0.93)
    risk = cur - sl
    t1   = cur + risk * 2
    t2   = cur + risk * 3
    t3   = cur + risk * 4
    rr   = (t2 - cur) / risk if risk > 0 else 0

    # ⑧ 손익비 (15점)
    add("손익비", 15 if rr >= 3 else 10 if rr >= 2 else 0, 15)

    # 신호 판정 (200일선 여부와 무관하게 점수 기준 적용)
    if score >= 90:
        signal  = "🚨 강력 매수"
        action  = "풀 포지션 진입 (자본의 10%)"
        pos_pct, max_loss_pct = 10, 2
    elif score >= 80:
        signal  = "✅ 매수"
        action  = "하프 포지션 진입 (자본의 5%)"
        pos_pct, max_loss_pct = 5, 1
    elif score >= 70:
        signal  = "⚠️ 관망"
        action  = "테스트 포지션 (자본의 2.5%) 또는 대기"
        pos_pct, max_loss_pct = 2.5, 0.5
    elif score >= 50:
        signal  = "⚠️ 관망"
        action  = "진입 금지 — 추이 관찰"
        pos_pct, max_loss_pct = 0, 0
    else:
        signal  = "💀 강력 매도" if score < 30 else "⛔ 매도"
        action  = "보유 중이면 청산 검토"
        pos_pct, max_loss_pct = 0, 0

    # 200일선 아래: 매수 신호면 포지션 절반 축소 + 경고
    if not tech["above_200"] and pos_pct > 0:
        pos_pct     = pos_pct / 2
        max_loss_pct = max_loss_pct / 2
        action += " ⚠️ 200일선 아래 — 포지션 절반 축소"

    pos_size = capital * pos_pct / 100
    shares   = int(pos_size / cur) if cur > 0 and pos_size > 0 else 0

    return dict(
        score=score, detail=detail,
        signal=signal, action=action,
        entry=cur, sl=round(sl, 2),
        t1=round(t1, 2), t2=round(t2, 2), t3=round(t3, 2),
        rr=round(rr, 1),
        trailing=round(tech["s10"], 2),
        pos_pct=pos_pct,
        pos_size=round(pos_size),
        shares=shares,
        max_loss=round(capital * max_loss_pct / 100),
        risk_per=round(risk, 2),
    )


# ──────────────────────────────────────────────────
# 리포트 출력
# ──────────────────────────────────────────────────

SEP  = "━" * 56
LINE = "─" * 56

def _fmt(v, dec=2):
    if v is None:
        return "N/A"
    return f"{v:,.{dec}f}"

def _pct(entry, v):
    return f"({(v - entry) / entry * 100:+.1f}%)"

def _wrap(text, width=56, indent=2):
    pad = " " * indent
    words = text.split()
    line, out = pad, []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = pad + w + " "
        else:
            line += w + " "
    if line.strip():
        out.append(line)
    return "\n".join(out)


def print_report(ticker, info, market, tech, risk, canslim, canslim_score, whale):
    name     = info.get("longName") or info.get("shortName") or ticker
    sector   = info.get("sector") or info.get("industry") or "N/A"
    currency = info.get("currency") or "USD"
    now_str  = datetime.now().strftime("%Y-%m-%d %H:%M")
    cur      = tech["cur"]

    print(f"\n{SEP}")
    print(f" 📊 [{name} / {ticker.upper()}] 심층 분석 리포트")
    print(SEP)
    print(f" 📅 분석일시 : {now_str}")
    print(f" 💹 현재가   : {_fmt(cur)} {currency}   섹터: {sector}")

    # ── 0단계: 시장 환경
    print(f"\n{LINE}")
    print(" 🌐 시장 환경 (Market Pulse)")
    print(LINE)
    print(f" 시장 상태       : {market['market_trend']}")
    print(f" VIX             : {market['vix']}  →  {market['vix_status']}")
    warn = "  ⚠️ 신규 매수 전면 금지" if market["dist_days"] >= 4 else ""
    print(f" Distribution Day: {market['dist_days']}회{warn}")

    # ── 최종 판결
    print(f"\n{LINE}")
    print(" ⚡ 최종 판결 (Signal)")
    print(LINE)
    print(f"  {risk['signal']}")
    print(f"  → {risk['action']}")

    print(f"\n 🎯 퀀트 스코어 : {risk['score']}/100점")
    print(f" {'─' * 44}")
    for k, (got, mx) in risk["detail"].items():
        filled = int(got / mx * 10) if mx > 0 else 0
        bar = "█" * filled + "░" * (10 - filled)
        print(f"  [{bar}] {k:<14s}: {got:2d}/{mx}점")

    # ── 기술적 분석
    print(f"\n{LINE}")
    print(" 📈 기술적 정밀 분석 (Technical Deep Dive)")
    print(LINE)

    s200_txt  = _fmt(tech["s200"]) if tech["s200"] else "형성 중"
    above_txt = "위 ✅" if tech["above_200"] else "아래 ⛔"

    print(f"\n 【추세 Trend】")
    print(f"  200일선  : {above_txt}  (SMA200: {s200_txt})")
    print(f"  Stage    : {tech['stage']}")
    print(f"  이평선   : {tech['ma_align']}")
    print(f"    SMA10={_fmt(tech['s10'])}  SMA20={_fmt(tech['s20'])}  SMA50={_fmt(tech['s50'])}  SMA200={s200_txt}")

    print(f"\n 【모멘텀 Momentum】")
    print(f"  RSI(14)   : {tech['rsi']}  →  {tech['rsi_status']}")
    print(f"  다이버전스 : {tech['divergence']}")
    print(f"  MACD      : {tech['macd_st']}")

    print(f"\n 【수급 Volume / OBV】")
    vol_tag = "✅ 유효 돌파" if tech["vol_ratio"] >= 150 else "⚠️ 거래량 부족" if tech["vol_ratio"] < 80 else ""
    print(f"  거래량    : 평균 대비 {tech['vol_ratio']}%  {vol_tag}")
    print(f"  OBV 추세  : {tech['obv_trend']}")
    obv_div = "Bullish Divergence ✅" if tech["obv_bull_div"] else "없음"
    print(f"  OBV 다이버전스: {obv_div}")

    print(f"\n 【볼린저 밴드】")
    if tech["bb_squeeze"]:
        bb_st = "🔵 수축 (Squeeze) — 폭발적 움직임 예고"
    elif tech["bb_expansion"]:
        bb_st = "🟠 확장 (Expansion) — 추세 돌파 중"
    else:
        bb_st = "보통"
    print(f"  밴드 상태 : {bb_st}")
    print(f"  %B 위치   : {tech['pct_b']:.3f}   (0=하단 / 0.5=중간 / 1=상단)")

    print(f"\n 【패턴 Pattern】")
    if tech["bb_expansion"] and tech["above_200"]:
        pat, pat_st = "볼린저 밴드 상향 돌파 (Expansion Breakout)", "완성 ✅"
    elif tech["bb_squeeze"]:
        pat, pat_st = "볼린저 밴드 수축 (Squeeze — 돌파 대기)", "형성 중 ⚠️"
    elif tech["stage_n"] == 2:
        pat, pat_st = "Stage 2 상승 추세 패턴", "진행 중 ✅"
    else:
        pat, pat_st = "명확한 패턴 미식별", "미확인"
    print(f"  식별 패턴 : {pat}")
    print(f"  완성도    : {pat_st}")
    print(f"  20일선    : {_fmt(tech['s20'])} {currency}")

    print(f"\n 【CANSLIM 체크】")
    labels = {
        "C": "최근 분기 EPS 25%+",
        "A": "연간 EPS 성장",
        "N": "신고가 / 신촉매",
        "S": "기관 수급",
        "L": "업종 선두 (RS 상위 20%)",
        "I": "기관 스폰서십",
        "M": "시장 방향",
    }
    for k, v in canslim.items():
        mk = "✅" if v else "❌"
        print(f"  {k}[{mk}] {labels.get(k, k)}")
    bonus = "+15점 보너스 ✅" if canslim_score >= 5 else f"({canslim_score}/7 충족 — 미달)"
    print(f"  충족: {canslim_score}/7항목  {bonus}")

    # ── 트레이딩 전략
    print(f"\n{LINE}")
    print(" ⚔️  0.1% 트레이딩 전략 (Trading Setup)")
    print(LINE)

    entry = risk["entry"]
    sl, t1, t2, t3, tr = risk["sl"], risk["t1"], risk["t2"], risk["t3"], risk["trailing"]

    print(f"  🎯 진입가   (Entry)     : {_fmt(entry):>14} {currency}")
    print(f"  💰 1차 목표  (Target 1) : {_fmt(t1):>14} {currency}  {_pct(entry, t1)}  → 33% 익절 + 손절→본전")
    print(f"  🚀 2차 목표  (Target 2) : {_fmt(t2):>14} {currency}  {_pct(entry, t2)}  → 33% 익절")
    print(f"  🌙 최종 목표 (Target 3) : {_fmt(t3):>14} {currency}  {_pct(entry, t3)}  → 트레일링 스탑")
    print(f"  🛑 손절가    (Stop)     : {_fmt(sl):>14} {currency}  {_pct(entry, sl)}")
    print(f"  🔔 트레일링 기준 (10SMA): {_fmt(tr):>14} {currency}")
    print(f"  📐 손익비    (R/R)      : 1 : {risk['rr']}")
    print(f"  📦 권장 포지션 비중     : 자본의 {risk['pos_pct']}%")
    if risk["shares"] > 0:
        print(f"  🔢 예상 주수            : {risk['shares']:,}주")
        print(f"  ⚡ 최대 허용 손실       : {risk['max_loss']:,} {currency}")

    # ── 세력 분석
    print(f"\n{LINE}")
    print(" 🐋 세력 분석 (Whale Tracker)")
    print(LINE)
    print("  감지된 시그니처:")
    for s in whale["sigs"]:
        print(f"    {s}")
    print(f"  세력 추정 의도 : {whale['intent']}")
    print(f"  주의 레벨      : {whale['alert']}")

    # ── Analyst's Insight
    print(f"\n{LINE}")
    print(" 💡 Analyst's Insight")
    print(LINE)

    sc = risk["score"]
    if not tech["above_200"]:
        insight = (f"'{name}'은(는) 200일선 아래에 있다. 차트가 아무리 매력적으로 "
                   f"보여도 신규 매수는 절대 금지. 현금이 최선의 포지션이다.")
    elif sc >= 90:
        insight = (f"'{name}'은(는) 퀀트 스코어 {sc}점의 최고 등급 셋업이다. "
                   f"시장이 지지하고 거래량이 확인된다면 주저 없이 진입하라. "
                   f"이런 기회는 1년에 몇 번 오지 않는다.")
    elif sc >= 80:
        insight = (f"'{name}'의 {sc}점 셋업은 충분히 매력적이다. 하프 포지션으로 "
                   f"진입하고, 1차 목표가 도달 시 손절가를 본전으로 올려 리스크 제로를 만들어라.")
    elif sc >= 70:
        insight = (f"'{name}'은(는) 가능성은 있으나 확신이 부족한 {sc}점 구간이다. "
                   f"소규모 테스트 포지션으로 시장 반응을 확인하고 전략을 재평가하라.")
    elif sc >= 50:
        insight = (f"'{name}'의 {sc}점은 진입 기준 미달이다. 현재 손익비가 불리하다 — "
                   f"인내심을 갖고 더 나은 타점을 기다려라. 기다림도 전략이다.")
    else:
        insight = (f"'{name}'의 {sc}점 신호는 명백한 청산 신호다. 보유 중이라면 "
                   f"손실이 커지기 전에 출구를 찾아라. 손절은 패배가 아닌 자본 보호다.")

    print(_wrap(insight, width=56))

    # ── 9계명 요약
    print(f"\n{LINE}")
    print(" 📌 불변의 트레이딩 법칙")
    print(LINE)
    rules = [
        "하락장에서는 현금이 최고의 포지션이다.",
        "손절은 감정이 아닌 가격으로 결정한다.",
        "한 종목에 총 자본의 2% 이상을 잃지 않는다.",
        "거래량 없는 상승은 믿지 않는다.",
        "퀀트 스코어 80점 미만인 자리는 진입하지 않는다.",
    ]
    for i, r in enumerate(rules, 1):
        print(f"  {i}. {r}")

    print(f"\n{SEP}")
    print(" ⚠️  본 분석은 투자 참고용이며, 투자 결정의 책임은")
    print("    전적으로 본인에게 있습니다.")
    print(f"{SEP}\n")


# ──────────────────────────────────────────────────
# 분석 실행
# ──────────────────────────────────────────────────

def run(ticker, capital):
    print(f"\n⏳ [{ticker}] 데이터 수집 중 ...")
    stock = yf.Ticker(ticker)
    df    = stock.history(period="2y")

    if df.empty:
        print(f"❌ '{ticker}' 데이터를 가져올 수 없습니다.")
        print("   한국주식 예: 005930.KS  000660.KS  035720.KQ")
        print("   미국주식 예: AAPL  TSLA  NVDA")
        return

    df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index

    info = {}
    try:
        info = stock.info or {}
    except Exception:
        pass

    print("⏳ 시장 환경 분석 중 ...")
    market = analyze_market()

    print("⏳ 기술적 지표 계산 중 ...")
    tech = analyze_technicals(df)

    print("⏳ CANSLIM 체크 중 ...")
    canslim, canslim_score = check_canslim(info, df, market["market_trend"])

    print("⏳ 퀀트 스코어 산출 중 ...")
    risk = score_and_risk(tech, canslim_score, capital)

    print("⏳ 세력 추적 중 ...")
    whale = analyze_whale(df, info)

    print("✅ 분석 완료!")
    print_report(ticker, info, market, tech, risk, canslim, canslim_score, whale)


# ──────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────

BANNER = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 📊  주식 심층 분석 프로그램
     CANSLIM + 퀀트 알파 프로토콜 (6단계)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 티커 입력 예시
   미국주식 : AAPL  TSLA  NVDA  MSFT  AMZN
   한국주식 : 005930.KS  000660.KS  035720.KQ
   지수     : ^GSPC  ^IXIC  ^KS11
   옵션     : --capital 50000000  (자본금 설정)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def main():
    parser = argparse.ArgumentParser(description="주식 심층 분석 — CANSLIM + 퀀트 알파")
    parser.add_argument("ticker", nargs="?", help="종목 티커")
    parser.add_argument("--capital", "-c", type=float, default=10_000_000,
                        help="투자 자본금 (기본: 10,000,000)")
    args = parser.parse_args()

    print(BANNER)
    capital = args.capital

    if args.ticker:
        run(args.ticker.upper(), capital)
        return

    while True:
        try:
            ticker = input("📌 종목 티커 입력 (종료: quit) > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n프로그램을 종료합니다.")
            break

        if not ticker:
            continue
        if ticker.lower() in ("quit", "exit", "q"):
            print("프로그램을 종료합니다.")
            break

        run(ticker.upper(), capital)


if __name__ == "__main__":
    main()
