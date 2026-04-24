import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


st.set_page_config(page_title="Trạm giao dịch Butterfly", layout="wide")


def apply_dark_theme() -> None:
    """Inject custom CSS for a professional dark terminal look."""
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #0B1220;
                color: #E5E7EB;
            }
            [data-testid="stSidebar"] {
                background-color: #111827;
                border-right: 1px solid #1F2937;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 1rem;
            }
            .stTabs [data-baseweb="tab"] {
                background-color: #111827;
                border: 1px solid #374151;
                border-radius: 8px 8px 0 0;
                color: #E5E7EB;
                padding: 0.5rem 1rem;
            }
            .stTabs [aria-selected="true"] {
                background-color: #1F2937 !important;
            }
            .metric-card {
                padding: 0.7rem 0.9rem;
                border: 1px solid #374151;
                border-radius: 10px;
                background: #0F172A;
                margin-bottom: 0.75rem;
            }
            .metric-label {
                color: #9CA3AF;
                font-size: 0.85rem;
                margin-bottom: 0.2rem;
            }
            .metric-value {
                color: #F9FAFB;
                font-size: 1.1rem;
                font-weight: 600;
            }
            .quote-strip {
                background: linear-gradient(135deg, #0F172A 0%, #111827 55%, #0B1220 100%);
                border: 1px solid #1F2937;
                border-radius: 12px;
                padding: 0.9rem 1rem;
                margin-bottom: 0.9rem;
            }
            .quote-row {
                display: flex;
                flex-wrap: wrap;
                gap: 1rem;
                align-items: center;
                justify-content: space-between;
            }
            .quote-main {
                font-size: 1.55rem;
                font-weight: 700;
                color: #F9FAFB;
            }
            .quote-sub {
                font-size: 0.85rem;
                color: #9CA3AF;
            }
            .pill {
                padding: 0.25rem 0.55rem;
                border-radius: 999px;
                font-size: 0.82rem;
                border: 1px solid #374151;
                background: #111827;
                color: #E5E7EB;
            }
            .order-ticket {
                background: #0F172A;
                border: 1px solid #1F2937;
                border-radius: 12px;
                padding: 0.8rem 1rem;
                margin: 0.7rem 0 0.2rem 0;
            }
            .ticket-title {
                font-size: 0.92rem;
                color: #9CA3AF;
                margin-bottom: 0.25rem;
            }
            .ticket-value {
                font-size: 1.25rem;
                color: #F9FAFB;
                font-weight: 700;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=300)
def load_ticker_data(symbol: str) -> tuple[float | None, list[str], str | None]:
    """Load current price and expiration dates using yfinance."""
    try:
        ticker_obj = yf.Ticker(symbol)
        hist = ticker_obj.history(period="5d", interval="1d")
        if hist.empty:
            return None, [], "Không lấy được dữ liệu giá cổ phiếu."
        spot = float(hist["Close"].dropna().iloc[-1])
        expirations = list(ticker_obj.options)
        return spot, expirations, None
    except Exception as exc:
        return None, [], f"Lỗi tải dữ liệu mã {symbol.upper()}: {exc}"


@st.cache_data(ttl=300)
def load_option_chain(symbol: str, expiry: str) -> tuple[pd.DataFrame | None, str | None]:
    """Load call option chain for a given expiration date."""
    try:
        calls = yf.Ticker(symbol).option_chain(expiry).calls.copy()
        if calls.empty:
            return None, "Không có dữ liệu quyền chọn Call cho ngày đáo hạn đã chọn."
        return calls, None
    except Exception as exc:
        return None, f"Lỗi tải chuỗi quyền chọn: {exc}"


@st.cache_data(ttl=300)
def load_price_history(symbol: str) -> tuple[pd.DataFrame | None, str | None]:
    """Load 6-month historical data for technical chart."""
    try:
        hist = yf.download(symbol, period="6mo", interval="1d", auto_adjust=False, progress=False)
        if hist.empty:
            return None, "Không có dữ liệu lịch sử 6 tháng."
        return hist, None
    except Exception as exc:
        return None, f"Lỗi tải biểu đồ kỹ thuật: {exc}"


@st.cache_data(ttl=300)
def load_quote_snapshot(symbol: str, spot_price: float) -> tuple[float, float, float]:
    """Load previous close / bid / ask for top quote strip."""
    prev_close = spot_price
    bid = max(0.01, spot_price - 0.01)
    ask = spot_price + 0.01
    try:
        ticker_obj = yf.Ticker(symbol)
        hist = ticker_obj.history(period="2d", interval="1d")
        if not hist.empty and len(hist["Close"].dropna()) >= 2:
            prev_close = float(hist["Close"].dropna().iloc[-2])
        fast = getattr(ticker_obj, "fast_info", None)
        if fast:
            fb = fast.get("bid")
            fa = fast.get("ask")
            if fb is not None and not pd.isna(fb) and fb > 0:
                bid = float(fb)
            if fa is not None and not pd.isna(fa) and fa > 0:
                ask = float(fa)
    except Exception:
        pass
    return prev_close, bid, ask


def option_leg_pnl(expiry_prices: np.ndarray, quantity: int, strike: float, premium: float) -> np.ndarray:
    """PnL at expiry for a call option leg."""
    intrinsic_value = np.maximum(0, expiry_prices - strike)
    if quantity > 0:
        return quantity * (intrinsic_value - premium)
    return abs(quantity) * (premium - intrinsic_value)


def strategy_pnl_at_price(
    price: float,
    leg1_qty: int,
    leg1_strike: float,
    leg1_premium: float,
    leg2_qty: int,
    leg2_strike: float,
    leg2_premium: float,
    leg3_qty: int,
    leg3_strike: float,
    leg3_premium: float,
) -> float:
    """Calculate total butterfly PnL at a single expiry price."""
    p = np.array([price], dtype=float)
    return float(
        option_leg_pnl(p, leg1_qty, leg1_strike, leg1_premium)[0]
        + option_leg_pnl(p, leg2_qty, leg2_strike, leg2_premium)[0]
        + option_leg_pnl(p, leg3_qty, leg3_strike, leg3_premium)[0]
    )


def build_scenario_candles(start_price: float, target_price: float, sessions: int) -> pd.DataFrame:
    """Create deterministic synthetic OHLC candles for illustrative scenarios."""
    sessions = max(4, int(sessions))
    dates = pd.date_range(pd.Timestamp.today().normalize(), periods=sessions, freq="B")
    trend = np.linspace(start_price, target_price, sessions)
    oscillation = np.sin(np.linspace(0, 3 * np.pi, sessions)) * max(0.1, abs(target_price - start_price) * 0.06)
    closes = np.maximum(0.01, trend + oscillation)
    opens = np.empty(sessions)
    opens[0] = start_price
    opens[1:] = closes[:-1]
    body = np.abs(closes - opens)
    wick = np.maximum(start_price * 0.0025, body * 0.45 + 0.03)
    highs = np.maximum(opens, closes) + wick
    lows = np.maximum(0.01, np.minimum(opens, closes) - wick)
    return pd.DataFrame({"Date": dates, "Open": opens, "High": highs, "Low": lows, "Close": closes})


def build_simulated_option_chain(spot_price: float, n_strikes: int = 31) -> pd.DataFrame:
    """Generate synthetic call option chain for demo when live data is sparse."""
    half = n_strikes // 2
    step = max(1.0, round(spot_price * 0.01))
    base = round(spot_price / step) * step
    strikes = np.array([base + (i - half) * step for i in range(n_strikes)], dtype=float)
    strikes = np.clip(strikes, 0.01, None)
    moneyness = (strikes - spot_price) / max(spot_price, 1e-6)

    # Smooth synthetic bid/ask curve around spot for presentation.
    mid = 2.8 * np.exp(-np.abs(moneyness) * 6.0) + np.maximum(0, (spot_price - strikes) * 0.14)
    spread = 0.08 + 0.18 * np.abs(moneyness)
    bid = np.maximum(0.01, mid - spread / 2)
    ask = np.maximum(bid + 0.01, mid + spread / 2)
    volume = np.maximum(1, (450 * np.exp(-np.abs(moneyness) * 4.5)).astype(int))
    iv = 0.18 + 0.35 * np.abs(moneyness)

    return pd.DataFrame(
        {
            "strike": strikes.round(2),
            "bid": bid.round(2),
            "ask": ask.round(2),
            "volume": volume,
            "impliedVolatility": iv.round(4),
        }
    )


def find_break_evens(prices: np.ndarray, pnl: np.ndarray) -> list[float]:
    """Find break-even points using interpolation around zero-crossings."""
    break_evens: list[float] = []
    for i in range(len(prices) - 1):
        y1, y2 = pnl[i], pnl[i + 1]
        if y1 == 0:
            break_evens.append(float(prices[i]))
        elif y1 * y2 < 0:
            x1, x2 = prices[i], prices[i + 1]
            x0 = x1 - y1 * (x2 - x1) / (y2 - y1)
            break_evens.append(float(x0))
    return sorted({round(point, 3) for point in break_evens})


def get_leg_premium(calls_df: pd.DataFrame, strike: float, quantity: int) -> float:
    """Use Ask for buy legs and Bid for sell legs."""
    row = calls_df.loc[np.isclose(calls_df["strike"], strike, rtol=0, atol=1e-6)]
    if row.empty:
        return 0.0
    record = row.iloc[0]
    if quantity > 0:
        value = record.get("ask", np.nan)
    else:
        value = record.get("bid", np.nan)
    if pd.isna(value):
        return 0.0
    return float(value)


apply_dark_theme()
st.title("Trạm giao dịch Butterfly Spread")
st.caption("Mô phỏng thực chiến chiến lược Butterfly bằng dữ liệu quyền chọn thực từ Yahoo Finance.")

with st.sidebar:
    st.header("Thiết lập mã & Lệnh")
    ticker = st.text_input("Mã cổ phiếu", value="SPY").strip().upper()

with st.spinner("Đang tải giá hiện tại và danh sách ngày đáo hạn..."):
    spot_price, expirations, ticker_error = load_ticker_data(ticker)

if ticker_error:
    st.error(ticker_error)
    st.stop()
if spot_price is None:
    st.error("Không xác định được giá hiện tại.")
    st.stop()
if not expirations:
    st.error(f"Mã {ticker} hiện không có dữ liệu Option trên Yahoo Finance.")
    st.stop()

with st.sidebar:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Giá hiện tại ({ticker})</div>
            <div class="metric-value">${spot_price:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_expiry = st.selectbox("Ngày đáo hạn", expirations, index=0)
    use_simulated_data = st.checkbox(
        "Dùng dữ liệu mô phỏng (Option Chain)",
        value=False,
        help="Bật khi dữ liệu thật quá thưa, nhiều giá 0 hoặc cần trình chiếu minh họa.",
    )

with st.spinner("Đang tải chuỗi quyền chọn..."):
    calls_df, chain_error = load_option_chain(ticker, selected_expiry)
if chain_error or calls_df is None:
    if use_simulated_data:
        calls_df = build_simulated_option_chain(spot_price)
        st.info("Đang dùng Option Chain mô phỏng do dữ liệu thực không sẵn sàng.")
    else:
        st.error(chain_error or "Không tải được chuỗi quyền chọn.")
        st.stop()

if calls_df is not None:
    required_cols = {"strike", "bid", "ask", "volume", "impliedVolatility"}
    missing_cols = required_cols.difference(set(calls_df.columns))
    if missing_cols:
        if use_simulated_data:
            calls_df = build_simulated_option_chain(spot_price)
            st.info("Option Chain thực thiếu cột cần thiết, đã chuyển sang dữ liệu mô phỏng.")
        else:
            st.error(f"Dữ liệu Option Chain thiếu cột: {', '.join(sorted(missing_cols))}.")
            st.stop()

calls_df = calls_df.dropna(subset=["strike"]).copy()
calls_df["strike"] = calls_df["strike"].astype(float)
calls_df = calls_df.sort_values("strike")

# Auto fallback: if chain has mostly zero quotes, simulated data is clearer for demos.
valid_quote_ratio = float(((calls_df["ask"] > 0) | (calls_df["bid"] > 0)).mean()) if len(calls_df) else 0.0
if (not use_simulated_data) and valid_quote_ratio < 0.25:
    calls_df = build_simulated_option_chain(spot_price)
    st.warning("Option Chain thực có thanh khoản thấp (nhiều giá 0). App tự chuyển sang dữ liệu mô phỏng để trực quan hơn.")

available_strikes = calls_df["strike"].tolist()

if len(available_strikes) < 3:
    st.error("Không đủ strike để xây dựng chiến lược Butterfly.")
    st.stop()

def nearest_index(values: list[float], target: float) -> int:
    return min(range(len(values)), key=lambda idx: abs(values[idx] - target))


lower_default = nearest_index(available_strikes, spot_price * 0.95)
body_default = nearest_index(available_strikes, spot_price)
upper_default = nearest_index(available_strikes, spot_price * 1.05)

with st.sidebar:
    st.subheader("Butterfly Legs")
    st.caption("Chân 1 = Mua 1 | Chân 2 = Bán 2 | Chân 3 = Mua 1")
    leg1_qty = 1
    leg2_qty = -2
    leg3_qty = 1

    leg1_strike = st.selectbox("Chân 1 (Mua 1) - Giá thực hiện", available_strikes, index=lower_default)
    leg2_strike = st.selectbox("Chân 2 (Bán 2) - Giá thực hiện", available_strikes, index=body_default)
    leg3_strike = st.selectbox("Chân 3 (Mua 1) - Giá thực hiện", available_strikes, index=upper_default)

    leg1_premium = get_leg_premium(calls_df, leg1_strike, leg1_qty)
    leg2_premium = get_leg_premium(calls_df, leg2_strike, leg2_qty)
    leg3_premium = get_leg_premium(calls_df, leg3_strike, leg3_qty)

    net_cashflow = (-leg1_premium * leg1_qty) + (leg2_premium * abs(leg2_qty)) + (-leg3_premium * leg3_qty)
    net_label = "Thu ròng (Net Credit)" if net_cashflow > 0 else "Chi ròng (Net Debit)"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Premium tự động từ Bid/Ask</div>
            <div class="metric-value">C1 Ask: {leg1_premium:.2f} | C2 Bid: {leg2_premium:.2f} | C3 Ask: {leg3_premium:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">{net_label}</div>
            <div class="metric-value">${abs(net_cashflow):,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # A balanced butterfly usually has equal wing widths around the body strike.
    lower_width = float(leg2_strike - leg1_strike)
    upper_width = float(leg3_strike - leg2_strike)
    if lower_width <= 0 or upper_width <= 0:
        st.warning("Cấu trúc strike chưa hợp lệ. Cần đảm bảo: Leg 1 < Leg 2 < Leg 3.")
    elif abs(lower_width - upper_width) > 1e-9:
        st.warning(
            f"Butterfly chưa cân: cánh dưới = {lower_width:.2f}, cánh trên = {upper_width:.2f}. "
            "Nên đặt cách đều tâm để mô phỏng Butterfly chuẩn."
        )
    else:
        st.success(f"Butterfly cân: mỗi cánh cách tâm {lower_width:.2f} điểm.")

prev_close, stock_bid, stock_ask = load_quote_snapshot(ticker, spot_price)
daily_change = spot_price - prev_close
daily_change_pct = (daily_change / prev_close * 100) if prev_close else 0.0
change_color = "#22C55E" if daily_change >= 0 else "#EF4444"
change_sign = "+" if daily_change >= 0 else ""

st.markdown(
    f"""
    <div class="quote-strip">
        <div class="quote-row">
            <div>
                <div class="quote-sub">{ticker} · Giao diện mô phỏng phong cách app</div>
                <div class="quote-main">${spot_price:,.2f}</div>
                <div style="color:{change_color}; font-weight:600;">
                    {change_sign}{daily_change:,.2f} ({change_sign}{daily_change_pct:.2f}%)
                </div>
            </div>
            <div style="display:flex; gap:0.55rem; flex-wrap:wrap;">
                <span class="pill">Bid: ${stock_bid:,.2f}</span>
                <span class="pill">Ask: ${stock_ask:,.2f}</span>
                <span class="pill">Đáo hạn: {selected_expiry}</span>
                <span class="pill">Bướm: {leg1_strike:.0f} / {leg2_strike:.0f} / {leg3_strike:.0f}</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# PnL simulation range centered around spot price
min_price = max(0.01, spot_price * 0.6)
max_price = spot_price * 1.4
expiry_prices = np.linspace(min_price, max_price, 500)

leg1_pnl = option_leg_pnl(expiry_prices, leg1_qty, leg1_strike, leg1_premium)
leg2_pnl = option_leg_pnl(expiry_prices, leg2_qty, leg2_strike, leg2_premium)
leg3_pnl = option_leg_pnl(expiry_prices, leg3_qty, leg3_strike, leg3_premium)
total_pnl = leg1_pnl + leg2_pnl + leg3_pnl

max_profit = float(np.max(total_pnl))
max_loss = float(np.min(total_pnl))
break_evens = find_break_evens(expiry_prices, total_pnl)

tab_guide, tab_tech, tab_chain, tab_payoff = st.tabs(
    ["Hướng dẫn sử dụng", "Biểu đồ Kỹ thuật", "Bảng giá Quyền chọn", "Biểu đồ Thực chiến Bướm"]
)

with tab_guide:
    st.markdown(
        """
### Hướng dẫn 3 bước để triển khai Butterfly Spread

**Bước 1 - Phân tích xu hướng:**  
Vào tab **Biểu đồ Kỹ thuật** để xem nến giá, SMA20/SMA50 và Khối lượng. Xác định vùng giá có khả năng "neo giá" (max pain) vào ngày đáo hạn.

**Bước 2 - Lập cấu trúc Butterfly:**  
Vào tab **Bảng giá Quyền chọn** để tìm strike phù hợp. Trên Sidebar:
- Đặt **Chân 2 (Bán 2)** tại mức strike dự đoán giá sẽ neo.
- Đặt **Chân 1 và Chân 3 (Mua)** cách đều tâm để tạo 2 cánh bảo vệ rủi ro.

**Bước 3 - Kiểm tra payoff thực chiến:**  
Vào tab **Biểu đồ Thực chiến Bướm** để đánh giá kịch bản lãi/lỗ, điểm hòa vốn, lãi tối đa và lỗ tối đa trước khi vào lệnh.
        """
    )

with tab_tech:
    st.subheader(f"{ticker} - Nến Nhật, SMA, Khối lượng (6 tháng)")
    with st.spinner("Đang tải dữ liệu kỹ thuật 6 tháng..."):
        hist_df, hist_error = load_price_history(ticker)
    if hist_error or hist_df is None:
        st.error(hist_error or "Không tải được dữ liệu lịch sử.")
    else:
        chart_df = hist_df.copy()
        chart_df["SMA20"] = chart_df["Close"].rolling(20).mean()
        chart_df["SMA50"] = chart_df["Close"].rolling(50).mean()

        tech_fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.72, 0.28],
            subplot_titles=("Nến Nhật + SMA20/SMA50", "Khối lượng"),
        )
        tech_fig.add_trace(
            go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open"],
                high=chart_df["High"],
                low=chart_df["Low"],
                close=chart_df["Close"],
                name="Giá",
            ),
            row=1,
            col=1,
        )
        tech_fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["SMA20"], mode="lines", name="SMA20", line=dict(color="#22D3EE")),
            row=1,
            col=1,
        )
        tech_fig.add_trace(
            go.Scatter(x=chart_df.index, y=chart_df["SMA50"], mode="lines", name="SMA50", line=dict(color="#F59E0B")),
            row=1,
            col=1,
        )
        tech_fig.add_trace(
            go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Khối lượng", marker_color="#64748B"),
            row=2,
            col=1,
        )
        tech_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1220",
            plot_bgcolor="#0B1220",
            xaxis_rangeslider_visible=False,
            margin=dict(l=25, r=25, t=40, b=20),
            legend=dict(orientation="h", y=1.02, x=0.01),
            height=700,
        )
        tech_fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
        tech_fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
        st.plotly_chart(tech_fig, use_container_width=True)

with tab_chain:
    st.subheader(f"Bảng quyền chọn Call - {ticker} | Đáo hạn: {selected_expiry}")
    lower_bound = spot_price * 0.9
    upper_bound = spot_price * 1.1
    display_df = calls_df.loc[
        (calls_df["strike"] >= lower_bound) & (calls_df["strike"] <= upper_bound),
        ["strike", "bid", "ask", "volume", "impliedVolatility"],
    ].copy()
    display_df.columns = ["Giá thực hiện", "Bid", "Ask", "Khối lượng", "Biến động hàm ý"]
    display_df = display_df.sort_values("Giá thực hiện")
    display_df["Spread"] = (display_df["Ask"] - display_df["Bid"]).round(3)

    selected_strikes = {float(leg1_strike), float(leg2_strike), float(leg3_strike)}

    def highlight_selected(row: pd.Series) -> list[str]:
        if float(row["Giá thực hiện"]) in selected_strikes:
            return ["background-color: rgba(34,197,94,0.20); font-weight: 600;"] * len(row)
        return [""] * len(row)

    def color_bid_ask(col: pd.Series) -> list[str]:
        if col.name == "Bid":
            return ["color: #22C55E; font-weight: 600;" for _ in col]
        if col.name == "Ask":
            return ["color: #EF4444; font-weight: 600;" for _ in col]
        return ["" for _ in col]

    left_col, right_col = st.columns([3, 2])
    with left_col:
        st.dataframe(
            display_df.style.apply(highlight_selected, axis=1).apply(color_bid_ask, axis=0),
            use_container_width=True,
            hide_index=True,
        )
    with right_col:
        st.markdown(
            f"""
            <div class="order-ticket">
                <div class="ticket-title">Phiếu lệnh nhanh</div>
                <div class="ticket-value">{ticker} CALL BUTTERFLY</div>
                <div class="quote-sub">+1 {leg1_strike:.0f}C / -2 {leg2_strike:.0f}C / +1 {leg3_strike:.0f}C</div>
                <div style="margin-top:0.45rem; color:#E5E7EB;">Giá vào lệnh mô phỏng: <b>${abs(net_cashflow):.2f}</b></div>
                <div class="quote-sub">{net_label}</div>
            </div>
        """,
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Chân lệnh": ["Chân 1", "Chân 2", "Chân 3"],
                    "Hành động": ["Mua", "Bán", "Mua"],
                    "Số lượng": [1, -2, 1],
                    "Strike": [leg1_strike, leg2_strike, leg3_strike],
                    "Premium": [leg1_premium, leg2_premium, leg3_premium],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    st.caption("Phong cách terminal: Bid xanh, Ask đỏ, và phiếu lệnh nổi bật để thao tác nhanh.")

with tab_payoff:
    st.subheader("Mô phỏng Payoff tại ngày đáo hạn")
    payoff_fig = go.Figure()
    payoff_fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", line_width=1)

    payoff_fig.add_trace(
        go.Scatter(
            x=expiry_prices,
            y=leg1_pnl,
            mode="lines",
            line=dict(color="rgba(59,130,246,0.6)", dash="dash"),
            name=f"Chân 1 Mua {leg1_strike:.2f}",
        )
    )
    payoff_fig.add_trace(
        go.Scatter(
            x=expiry_prices,
            y=leg2_pnl,
            mode="lines",
            line=dict(color="rgba(168,85,247,0.6)", dash="dash"),
            name=f"Chân 2 Bán 2 {leg2_strike:.2f}",
        )
    )
    payoff_fig.add_trace(
        go.Scatter(
            x=expiry_prices,
            y=leg3_pnl,
            mode="lines",
            line=dict(color="rgba(245,158,11,0.6)", dash="dash"),
            name=f"Chân 3 Mua {leg3_strike:.2f}",
        )
    )

    payoff_fig.add_trace(
        go.Scatter(
            x=expiry_prices,
            y=np.where(total_pnl >= 0, total_pnl, np.nan),
            mode="lines",
            line=dict(color="rgba(16,185,129,0.45)", width=0),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.20)",
            showlegend=True,
            name="Vùng lãi",
        )
    )
    payoff_fig.add_trace(
        go.Scatter(
            x=expiry_prices,
            y=np.where(total_pnl < 0, total_pnl, np.nan),
            mode="lines",
            line=dict(color="rgba(239,68,68,0.45)", width=0),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.20)",
            showlegend=True,
            name="Vùng lỗ",
        )
    )
    payoff_fig.add_trace(
        go.Scatter(
            x=expiry_prices,
            y=total_pnl,
            mode="lines",
            line=dict(color="#22C55E", width=4),
            name="Tổng PnL Butterfly",
            hovertemplate="Giá đáo hạn: %{x:.2f}<br>PnL: %{y:.2f}<extra></extra>",
        )
    )

    max_profit_idx = int(np.argmax(total_pnl))
    max_loss_idx = int(np.argmin(total_pnl))
    payoff_fig.add_trace(
        go.Scatter(
            x=[float(expiry_prices[max_profit_idx])],
            y=[float(total_pnl[max_profit_idx])],
            mode="markers+text",
            marker=dict(size=11, color="#10B981"),
            text=["Lãi tối đa"],
            textposition="top center",
            name="Lãi tối đa",
        )
    )
    payoff_fig.add_trace(
        go.Scatter(
            x=[float(expiry_prices[max_loss_idx])],
            y=[float(total_pnl[max_loss_idx])],
            mode="markers+text",
            marker=dict(size=11, color="#EF4444"),
            text=["Lỗ tối đa"],
            textposition="bottom center",
            name="Lỗ tối đa",
        )
    )
    if break_evens:
        payoff_fig.add_trace(
            go.Scatter(
                x=break_evens,
                y=[0] * len(break_evens),
                mode="markers+text",
                marker=dict(size=10, symbol="diamond", color="#FBBF24"),
                text=["Hòa vốn"] * len(break_evens),
                textposition="top center",
                name="Hòa vốn",
            )
        )

    payoff_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        xaxis_title="Giá cổ phiếu khi đáo hạn",
        yaxis_title="Lợi nhuận / Thua lỗ",
        margin=dict(l=25, r=25, t=30, b=25),
        legend=dict(orientation="h", y=1.02, x=0.01),
        hovermode="x unified",
    )
    payoff_fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    payoff_fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    st.plotly_chart(payoff_fig, use_container_width=True)

    be_text = ", ".join(f"{x:.2f}" for x in break_evens) if break_evens else "Không có"
    col1, col2, col3 = st.columns(3)
    col1.metric("Điểm hòa vốn", be_text)
    col2.metric("Lãi tối đa", f"{max_profit:.2f}")
    col3.metric("Lỗ tối đa", f"{max_loss:.2f}")

    st.markdown("### Kịch bản giả định khi thị trường biến động")
    in_col1, in_col2, in_col3, in_col4 = st.columns(4)
    with in_col1:
        so_phien_gia_dinh = st.number_input("Số phiên mô phỏng", min_value=4, max_value=60, value=12, step=1)
    with in_col2:
        ty_le_tang = st.number_input("% tăng giả định", min_value=0.0, max_value=100.0, value=6.0, step=0.5)
    with in_col3:
        ty_le_giam = st.number_input("% giảm giả định", min_value=0.0, max_value=100.0, value=6.0, step=0.5)
    with in_col4:
        huong_mo_phong = st.selectbox("Hướng mô phỏng nến", ["Tăng", "Giảm"], index=0)

    gia_tang = spot_price * (1 + ty_le_tang / 100)
    gia_giam = spot_price * (1 - ty_le_giam / 100)
    gia_di_ngang = spot_price

    pnl_tang = strategy_pnl_at_price(
        gia_tang, leg1_qty, leg1_strike, leg1_premium, leg2_qty, leg2_strike, leg2_premium, leg3_qty, leg3_strike, leg3_premium
    )
    pnl_giam = strategy_pnl_at_price(
        gia_giam, leg1_qty, leg1_strike, leg1_premium, leg2_qty, leg2_strike, leg2_premium, leg3_qty, leg3_strike, leg3_premium
    )
    pnl_di_ngang = strategy_pnl_at_price(
        gia_di_ngang, leg1_qty, leg1_strike, leg1_premium, leg2_qty, leg2_strike, leg2_premium, leg3_qty, leg3_strike, leg3_premium
    )

    scenario_df = pd.DataFrame(
        {
            "Kịch bản": ["Tăng", "Giảm", "Đi ngang"],
            "Biến động (%)": [ty_le_tang, -ty_le_giam, 0.0],
            "Giá mục tiêu": [gia_tang, gia_giam, gia_di_ngang],
            "PnL ước tính": [pnl_tang, pnl_giam, pnl_di_ngang],
        }
    )
    st.dataframe(
        scenario_df.style.format({"Biến động (%)": "{:+.2f}", "Giá mục tiêu": "{:.2f}", "PnL ước tính": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    target_price = gia_tang if huong_mo_phong == "Tăng" else gia_giam
    simulated_candles = build_scenario_candles(spot_price, target_price, int(so_phien_gia_dinh))
    scenario_fig = go.Figure()
    scenario_fig.add_trace(
        go.Candlestick(
            x=simulated_candles["Date"],
            open=simulated_candles["Open"],
            high=simulated_candles["High"],
            low=simulated_candles["Low"],
            close=simulated_candles["Close"],
            name="Nến mô phỏng",
        )
    )
    scenario_fig.add_hline(y=leg2_strike, line_dash="dot", line_color="#FBBF24", annotation_text="Tâm bướm")
    scenario_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        xaxis_title="Phiên giao dịch giả định",
        yaxis_title="Giá cổ phiếu",
        xaxis_rangeslider_visible=False,
        margin=dict(l=25, r=25, t=20, b=20),
        height=420,
    )
    st.plotly_chart(scenario_fig, use_container_width=True)

    # Extra visual: PnL bar chart at key price levels.
    key_prices = [
        ("Cánh dưới", float(leg1_strike)),
        ("Tâm bướm", float(leg2_strike)),
        ("Cánh trên", float(leg3_strike)),
        ("Giá hiện tại", float(spot_price)),
    ]
    if break_evens:
        key_prices.append(("Hòa vốn 1", float(break_evens[0])))
        if len(break_evens) > 1:
            key_prices.append(("Hòa vốn 2", float(break_evens[1])))

    key_labels = [item[0] for item in key_prices]
    key_values = [item[1] for item in key_prices]
    key_pnls = [
        float(option_leg_pnl(np.array([p]), leg1_qty, leg1_strike, leg1_premium)[0])
        + float(option_leg_pnl(np.array([p]), leg2_qty, leg2_strike, leg2_premium)[0])
        + float(option_leg_pnl(np.array([p]), leg3_qty, leg3_strike, leg3_premium)[0])
        for p in key_values
    ]
    bar_colors = ["#10B981" if pnl >= 0 else "#EF4444" for pnl in key_pnls]

    st.markdown("### Đồ thị bổ sung: PnL tại các mốc giá quan trọng")
    key_fig = go.Figure(
        data=[
            go.Bar(
                x=key_labels,
                y=key_pnls,
                marker_color=bar_colors,
                text=[f"{v:.2f}" for v in key_pnls],
                textposition="outside",
                hovertemplate="%{x}<br>PnL: %{y:.2f}<extra></extra>",
                name="PnL theo mốc giá",
            )
        ]
    )
    key_fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
    key_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        xaxis_title="Mốc giá tham chiếu",
        yaxis_title="Lợi nhuận / Thua lỗ",
        margin=dict(l=25, r=25, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(key_fig, use_container_width=True)