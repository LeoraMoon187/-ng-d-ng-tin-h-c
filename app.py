import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from io import StringIO
import streamlit as st
import yfinance as yf
from scipy.signal import argrelextrema
from scipy.stats import norm


st.set_page_config(page_title="Phòng thí nghiệm định giá quyền chọn BSM", layout="wide")


def apply_dark_theme() -> None:
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
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
                color: #FFFFFF !important;
            }
            .hero {
                border: 1px solid #334155;
                border-radius: 12px;
                background: linear-gradient(135deg, #111827 0%, #0F172A 60%, #0B1220 100%);
                padding: 0.9rem 1rem;
                margin-bottom: 0.8rem;
                color: #F8FAFC;
                font-weight: 600;
                font-size: 1rem;
            }
            .premium-box {
                border: 1px solid #334155;
                border-radius: 12px;
                background: #0F172A;
                padding: 0.9rem 1rem;
                margin-bottom: 0.75rem;
            }
            .premium-label {
                color: #94A3B8;
                font-size: 0.9rem;
                margin-bottom: 0.2rem;
            }
            .premium-value {
                color: #A3E635;
                font-size: 1.7rem;
                font-weight: 700;
            }
            [data-testid="stMetricLabel"],
            [data-testid="stMetricLabel"] p,
            [data-testid="stMetricValue"] {
                color: #FFFFFF !important;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                border-bottom: 2px solid #A3E635;
                color: #A3E635 !important;
            }
            .info-box {
                border: 1px solid #334155;
                border-radius: 12px;
                background: #0F172A;
                padding: 0.9rem 1rem;
                margin-top: 0.75rem;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )


def black_scholes_calc(
    s: float, k: float, t_years: float, r: float, sigma: float, option_type: str
) -> dict[str, float]:
    """Calculate BSM price and core Greeks (Delta, Gamma)."""
    t_eff = max(t_years, 1e-10)
    sigma_eff = max(sigma, 1e-10)
    d1 = (np.log(s / k) + (r + 0.5 * sigma_eff**2) * t_eff) / (sigma_eff * np.sqrt(t_eff))
    d2 = d1 - sigma_eff * np.sqrt(t_eff)

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)

    call_price = s * nd1 - k * np.exp(-r * t_eff) * nd2
    put_price = k * np.exp(-r * t_eff) * norm.cdf(-d2) - s * norm.cdf(-d1)

    if option_type == "Call":
        price = call_price
        delta = nd1
    else:
        price = put_price
        delta = nd1 - 1

    gamma = pdf_d1 / (s * sigma_eff * np.sqrt(t_eff))

    return {
        "d1": float(d1),
        "d2": float(d2),
        "Nd1": float(nd1),
        "Nd2": float(nd2),
        "call": float(call_price),
        "put": float(put_price),
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
    }


def simulate_gbm_paths(
    s0: float, t_years: float, r: float, sigma: float, n_steps: int = 120, n_paths: int = 10, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GBM price paths."""
    np.random.seed(seed)
    dt = t_years / n_steps
    times = np.linspace(0, t_years, n_steps + 1)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = s0
    for i in range(1, n_steps + 1):
        z = np.random.normal(size=n_paths)
        paths[:, i] = paths[:, i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return times, paths


def simulate_gbm_terminal_prices(
    s0: float, t_years: float, r: float, sigma: float, n_steps: int = 252, n_paths: int = 10_000, seed: int = 42
) -> np.ndarray:
    """Simulate terminal prices S_T under a GBM process."""
    np.random.seed(seed)
    dt = t_years / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    shocks = np.random.normal(size=(n_paths, n_steps))
    log_growth = np.cumsum(drift + diffusion * shocks, axis=1)
    return s0 * np.exp(log_growth[:, -1])


@st.cache_data(ttl=900)
def load_latest_close(symbol: str) -> tuple[float | None, str | None]:
    """Load latest closing price for a ticker."""
    try:
        hist = yf.download(symbol, period="1mo", interval="1d", auto_adjust=True, progress=False)
        if hist.empty or "Close" not in hist:
            return None, f"Không tải được giá gần nhất cho mã {symbol}."
        close_series = hist["Close"]
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        close_series = close_series.dropna()
        if close_series.empty:
            return None, f"Không có dữ liệu giá đóng cửa cho mã {symbol}."
        return float(close_series.iloc[-1]), None
    except Exception as exc:
        return None, f"Lỗi tải dữ liệu {symbol}: {exc}"


# ===============================
# BACKTEST CALL OPTION ASML.AS
# ===============================

def _flatten_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa cột dữ liệu yfinance nếu trả về MultiIndex."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=3600)
def load_backtest_price_data(
    symbol: str,
    start_date: pd.Timestamp,
    expiration_date: pd.Timestamp,
    lookback_days: int = 252,
    post_expiration_days: int = 10,
) -> tuple[pd.DataFrame | None, str | None]:
    """Tải giá OHLC đủ dài để tính volatility lịch sử và vẽ chart đến sau đáo hạn."""
    try:
        download_start = start_date - pd.Timedelta(days=max(lookback_days * 2, 370))
        download_end = expiration_date + pd.Timedelta(days=post_expiration_days + 14)
        data = yf.download(
            symbol,
            start=download_start.strftime("%Y-%m-%d"),
            end=download_end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
        )
        data = _flatten_yfinance_columns(data)
        if data.empty:
            return None, f"Không tải được dữ liệu giá cho mã {symbol}."
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return None, f"Dữ liệu thiếu cột: {', '.join(missing_cols)}."
        data = data.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce").fillna(0.0)
        data.index = pd.to_datetime(data.index).tz_localize(None)
        return data, None
    except Exception as exc:
        return None, f"Lỗi tải dữ liệu giá {symbol}: {exc}"


def nearest_trading_date(data: pd.DataFrame, target_date: pd.Timestamp, direction: str = "forward") -> pd.Timestamp:
    """Lấy ngày giao dịch gần nhất theo hướng forward/backward."""
    target_date = pd.Timestamp(target_date).normalize()
    if direction == "forward":
        valid_dates = data.index[data.index >= target_date]
        if len(valid_dates) == 0:
            raise ValueError(f"Không có ngày giao dịch sau {target_date.date()}.")
        return pd.Timestamp(valid_dates[0])
    valid_dates = data.index[data.index <= target_date]
    if len(valid_dates) == 0:
        raise ValueError(f"Không có ngày giao dịch trước {target_date.date()}.")
    return pd.Timestamp(valid_dates[-1])


def calculate_historical_volatility_from_close(
    data: pd.DataFrame,
    valuation_date: pd.Timestamp,
    lookback_days: int = 252,
    trading_days_per_year: int = 252,
) -> float:
    """Tính historical volatility annualized từ log-return trước ngày định giá."""
    hist = data.loc[data.index < valuation_date].tail(lookback_days).copy()
    if len(hist) < 30:
        raise ValueError("Không đủ dữ liệu lịch sử để tính historical volatility.")
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    sigma = float(log_returns.std() * np.sqrt(trading_days_per_year))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Historical volatility không hợp lệ.")
    return sigma


@st.cache_data(ttl=24 * 3600)
def load_ecb_estr_rate(valuation_date: pd.Timestamp, fallback_rate: float = 0.04) -> tuple[float, str]:
    """Lấy lãi suất €STR từ ECB quanh ngày định giá. Nếu lỗi, dùng fallback."""
    start_period = (pd.Timestamp(valuation_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end_period = pd.Timestamp(valuation_date).strftime("%Y-%m-%d")
    urls = [
        "https://data-api.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT",
        "https://sdw-wsrest.ecb.europa.eu/service/data/EST/B.EU000A2X2A25.WT",
    ]
    headers = {"Accept": "text/csv", "User-Agent": "Mozilla/5.0 option-backtest-streamlit"}
    for url in urls:
        try:
            resp = requests.get(
                url,
                params={"startPeriod": start_period, "endPeriod": end_period, "detail": "dataonly"},
                headers=headers,
                timeout=12,
            )
            if resp.status_code != 200 or not resp.text.strip():
                continue
            ecb_df = pd.read_csv(StringIO(resp.text))
            if ecb_df.empty:
                continue
            value_col = "OBS_VALUE" if "OBS_VALUE" in ecb_df.columns else None
            if value_col is None:
                numeric_cols = ecb_df.select_dtypes(include=[np.number]).columns.tolist()
                value_col = numeric_cols[-1] if numeric_cols else None
            if value_col is None:
                continue
            series = pd.to_numeric(ecb_df[value_col], errors="coerce").dropna()
            if series.empty:
                continue
            rate = float(series.iloc[-1]) / 100.0
            if -0.05 <= rate <= 0.20:
                return rate, "ECB €STR"
        except Exception:
            continue
    return float(fallback_rate), "Fallback thủ công"


def suggest_call_strike_expected_move(s0: float, sigma: float, t_years: float, std_mult: float = 1.0, rounding_base: float = 5.0) -> float:
    """Đề xuất K = S0 + expected move, làm tròn theo bội số rounding_base."""
    expected_move = s0 * sigma * np.sqrt(max(t_years, 1e-10)) * std_mult
    raw_k = s0 + expected_move
    return float(max(rounding_base, round(raw_k / rounding_base) * rounding_base))


def build_asml_call_backtest(
    symbol: str,
    start_date: pd.Timestamp,
    tenor_days: int,
    lookback_days: int,
    std_mult: float,
    rounding_base: float,
    fallback_rate: float,
) -> tuple[dict[str, float | str | pd.Timestamp], pd.DataFrame]:
    """Chạy backtest Call Option ASML theo BSM và trả về result + dữ liệu giá."""
    requested_start = pd.Timestamp(start_date).normalize()
    requested_expiry = requested_start + pd.Timedelta(days=int(tenor_days))
    data, err = load_backtest_price_data(symbol, requested_start, requested_expiry, lookback_days, 10)
    if err or data is None:
        raise ValueError(err or "Không có dữ liệu giá.")

    start_actual = nearest_trading_date(data, requested_start, "forward")
    expiry_actual = nearest_trading_date(data, requested_expiry, "forward")
    s0 = float(data.loc[start_actual, "Close"])
    st_expiry = float(data.loc[expiry_actual, "Close"])
    t_years = max((expiry_actual - start_actual).days / 365.0, 1e-10)
    sigma_hist = calculate_historical_volatility_from_close(data, start_actual, lookback_days)
    rate, rate_source = load_ecb_estr_rate(start_actual, fallback_rate)
    k_auto = suggest_call_strike_expected_move(s0, sigma_hist, t_years, std_mult, rounding_base)
    bs_res = black_scholes_calc(s0, k_auto, t_years, rate, sigma_hist, "Call")
    payoff = max(st_expiry - k_auto, 0.0)
    pnl = payoff - bs_res["price"]
    pnl_pct = pnl / bs_res["price"] if bs_res["price"] > 0 else np.nan
    result_dict = {
        "symbol": symbol,
        "requested_start": requested_start,
        "requested_expiry": requested_expiry,
        "start_actual": start_actual,
        "expiry_actual": expiry_actual,
        "s0": s0,
        "st_expiry": st_expiry,
        "k": k_auto,
        "t_years": t_years,
        "sigma_hist": sigma_hist,
        "r": rate,
        "rate_source": rate_source,
        "d1": bs_res["d1"],
        "d2": bs_res["d2"],
        "call_price": bs_res["price"],
        "payoff": payoff,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
    }
    return result_dict, data


def build_backtest_candlestick_figure(data: pd.DataFrame, result_dict: dict, post_expiration_days: int = 10) -> go.Figure:
    """Tạo biểu đồ nến cho backtest Call Option, kèm S0, K và annotation."""
    chart_start = result_dict["start_actual"]
    chart_end = result_dict["expiry_actual"] + pd.Timedelta(days=post_expiration_days)
    chart_data = data.loc[(data.index >= chart_start) & (data.index <= chart_end)].copy()
    if chart_data.empty:
        raise ValueError("Không có dữ liệu để vẽ biểu đồ nến.")

    fig = go.Figure()

    # --- BIỂU ĐỒ NẾN CHÍNH ---
    # Có thể đổi màu nến tăng/giảm tại increasing_line_color và decreasing_line_color.
    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name=f"{result_dict['symbol']} OHLC",
            increasing_line_color="#22C55E",
            decreasing_line_color="#EF4444",
            increasing_fillcolor="#22C55E",
            decreasing_fillcolor="#EF4444",
        )
    )

    # --- ĐƯỜNG NGANG S0 ---
    # Thể hiện giá cổ phiếu tại ngày bắt đầu mô phỏng.
    fig.add_hline(
        y=float(result_dict["s0"]),
        line_width=2,
        line_dash="solid",
        line_color="#38BDF8",
        annotation_text=f"S0 = {float(result_dict['s0']):.2f}",
        annotation_position="bottom right",
    )

    # --- ĐƯỜNG NGANG K ---
    # Đây là giá thực hiện mục tiêu, đề xuất theo Expected Move +1σ.
    # Dùng nét đứt màu xanh lá để dễ phân biệt với S0.
    fig.add_hline(
        y=float(result_dict["k"]),
        line_width=3,
        line_dash="dash",
        line_color="#A3E635",
        annotation_text=f"K mục tiêu = {float(result_dict['k']):.2f}",
        annotation_position="top right",
    )

    y_max = float(chart_data["High"].max())
    y_min = float(chart_data["Low"].min())
    y_ann = y_max + max(y_max - y_min, 1.0) * 0.05

    # --- MARKER NGÀY BẮT ĐẦU ---
    # Đánh dấu thời điểm mua quyền chọn / bắt đầu định giá.
    fig.add_vline(x=result_dict["start_actual"], line_width=2, line_dash="dot", line_color="#FBBF24")
    fig.add_annotation(
        x=result_dict["start_actual"],
        y=y_ann,
        text=f"Bắt đầu<br>{result_dict['start_actual'].date()}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-45,
        bgcolor="#111827",
        bordercolor="#FBBF24",
        font=dict(color="#F9FAFB", size=12),
    )

    # --- MARKER NGÀY ĐÁO HẠN ---
    # Đánh dấu ngày xác định payoff cuối cùng của Call Option.
    fig.add_vline(x=result_dict["expiry_actual"], line_width=2, line_dash="dot", line_color="#A78BFA")
    fig.add_annotation(
        x=result_dict["expiry_actual"],
        y=y_ann,
        text=f"Đáo hạn<br>{result_dict['expiry_actual'].date()}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-45,
        bgcolor="#111827",
        bordercolor="#A78BFA",
        font=dict(color="#F9FAFB", size=12),
    )

    fig.update_layout(
        title=(
            f"Backtest Call Option {result_dict['symbol']} | "
            f"S0={float(result_dict['s0']):.2f}, K={float(result_dict['k']):.2f}, "
            f"ST={float(result_dict['st_expiry']):.2f}"
        ),
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        xaxis_title="Ngày",
        yaxis_title="Giá cổ phiếu",
        height=650,
        margin=dict(l=25, r=25, t=70, b=25),
        hovermode="x unified",
    )
    # Tắt range slider cho giao diện gọn; đổi thành True nếu muốn kéo vùng thời gian.
    fig.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return fig


def format_backtest_report(result_dict: dict) -> str:
    """Tạo báo cáo kiểm thử dạng text để đặt ngay bên dưới chart."""
    status = "ITM - Call có giá trị khi đáo hạn" if result_dict["payoff"] > 0 else "OTM - Call hết hạn vô giá trị"
    lines = [
        "=" * 78,
        "BÁO CÁO KIỂM THỬ CALL OPTION ASML THEO BLACK-SCHOLES",
        "=" * 78,
        f"Mã cổ phiếu                         : {result_dict['symbol']}",
        f"Ngày bắt đầu yêu cầu                : {result_dict['requested_start'].date()}",
        f"Ngày bắt đầu thực tế                : {result_dict['start_actual'].date()}",
        f"Ngày đáo hạn yêu cầu                : {result_dict['requested_expiry'].date()}",
        f"Ngày đáo hạn thực tế                : {result_dict['expiry_actual'].date()}",
        f"Thời gian đáo hạn T                 : {float(result_dict['t_years']):.6f} năm",
        "-" * 78,
        f"Giá cổ phiếu ban đầu S0             : {float(result_dict['s0']):,.4f}",
        f"Giá cổ phiếu tại đáo hạn ST         : {float(result_dict['st_expiry']):,.4f}",
        f"Giá thực hiện đề xuất K             : {float(result_dict['k']):,.4f}",
        "-" * 78,
        f"Historical Volatility sigma         : {float(result_dict['sigma_hist']):.4%}",
        f"Lãi suất phi rủi ro r               : {float(result_dict['r']):.4%} ({result_dict['rate_source']})",
        f"d1                                  : {float(result_dict['d1']):.6f}",
        f"d2                                  : {float(result_dict['d2']):.6f}",
        "-" * 78,
        f"Giá Call theo Black-Scholes         : {float(result_dict['call_price']):,.4f}",
        f"Payoff tại đáo hạn max(ST-K,0)      : {float(result_dict['payoff']):,.4f}",
        f"Lãi / Lỗ nếu giữ đến đáo hạn        : {float(result_dict['pnl']):,.4f}",
        f"Tỷ suất trên premium                : {float(result_dict['pnl_pct']):.2%}",
        f"Trạng thái đáo hạn                  : {status}",
        "=" * 78,
    ]
    return "\n".join(lines)



# ===============================
# BACKTEST NÂNG CAO: CANDLESTICK + VOLUME + TAM GIÁC CÂN
# ===============================

def _choose_two_decreasing_highs(maxima_df: pd.DataFrame) -> pd.DataFrame:
    """Chọn 2 đỉnh thấp dần gần ngày định giá nhất để tạo đường cản trên."""
    if len(maxima_df) < 2:
        return pd.DataFrame()
    rows = maxima_df.sort_index()
    candidates = []
    for i in range(len(rows) - 1):
        for j in range(i + 1, len(rows)):
            if float(rows.iloc[j]["High"]) < float(rows.iloc[i]["High"]):
                candidates.append((rows.index[i], rows.index[j]))
    if not candidates:
        return pd.DataFrame()
    idx1, idx2 = candidates[-1]
    return rows.loc[[idx1, idx2]]


def _choose_two_increasing_lows(minima_df: pd.DataFrame) -> pd.DataFrame:
    """Chọn 2 đáy cao dần gần ngày định giá nhất để tạo đường hỗ trợ dưới."""
    if len(minima_df) < 2:
        return pd.DataFrame()
    rows = minima_df.sort_index()
    candidates = []
    for i in range(len(rows) - 1):
        for j in range(i + 1, len(rows)):
            if float(rows.iloc[j]["Low"]) > float(rows.iloc[i]["Low"]):
                candidates.append((rows.index[i], rows.index[j]))
    if not candidates:
        return pd.DataFrame()
    idx1, idx2 = candidates[-1]
    return rows.loc[[idx1, idx2]]


def _fit_line_from_two_points(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """Trả về hệ số a, b của đường thẳng y = a*x + b."""
    if abs(x2 - x1) < 1e-10:
        raise ValueError("Không thể fit trendline khi hai điểm có cùng hoành độ.")
    a = (float(y2) - float(y1)) / (float(x2) - float(x1))
    b = float(y1) - a * float(x1)
    return float(a), float(b)


def detect_symmetrical_triangle(data: pd.DataFrame, valuation_date: pd.Timestamp, lookback_days: int = 62, order: int = 5) -> dict:
    """
    Nhận diện mô hình Tam giác cân trước ngày định giá.

    Thuật toán tìm đỉnh/đáy:
    - Lấy dữ liệu trong khoảng lookback_days trước ngày định giá.
    - Dùng scipy.signal.argrelextrema để tìm cực trị cục bộ:
      + Local maxima: điểm High cao hơn các điểm lân cận trong cửa sổ order.
      + Local minima: điểm Low thấp hơn các điểm lân cận trong cửa sổ order.
    - Từ các cực trị đó, chọn 2 đỉnh thấp dần (Lower Highs) và 2 đáy cao dần (Higher Lows).
    - Fit 2 đường thẳng: đường cản trên phải dốc xuống, đường hỗ trợ dưới phải dốc lên.
    - Nếu 2 đường hội tụ, trả về thông tin trendline để vẽ lên biểu đồ.
    """
    valuation_date = pd.Timestamp(valuation_date).normalize()
    pattern_data = data.loc[(data.index < valuation_date) & (data.index >= valuation_date - pd.Timedelta(days=int(lookback_days)))].copy()
    if len(pattern_data) < max(20, int(order) * 4):
        return {"found": False, "reason": "Không đủ dữ liệu trước ngày định giá để nhận diện tam giác."}

    highs = pattern_data["High"].to_numpy(dtype=float)
    lows = pattern_data["Low"].to_numpy(dtype=float)
    maxima_idx = argrelextrema(highs, np.greater_equal, order=int(order))[0]
    minima_idx = argrelextrema(lows, np.less_equal, order=int(order))[0]
    maxima_df = pattern_data.iloc[maxima_idx][["High"]].copy()
    minima_df = pattern_data.iloc[minima_idx][["Low"]].copy()

    selected_highs = _choose_two_decreasing_highs(maxima_df)
    selected_lows = _choose_two_increasing_lows(minima_df)
    if selected_highs.empty or selected_lows.empty:
        return {"found": False, "reason": "Không tìm được đủ 2 đỉnh thấp dần và 2 đáy cao dần.", "maxima": maxima_df, "minima": minima_df}

    all_dates = pd.Series(np.arange(len(data)), index=data.index)
    h_dates = selected_highs.index
    l_dates = selected_lows.index
    xh1, xh2 = float(all_dates.loc[h_dates[0]]), float(all_dates.loc[h_dates[1]])
    yh1, yh2 = float(selected_highs.iloc[0]["High"]), float(selected_highs.iloc[1]["High"])
    xl1, xl2 = float(all_dates.loc[l_dates[0]]), float(all_dates.loc[l_dates[1]])
    yl1, yl2 = float(selected_lows.iloc[0]["Low"]), float(selected_lows.iloc[1]["Low"])

    upper_a, upper_b = _fit_line_from_two_points(xh1, yh1, xh2, yh2)
    lower_a, lower_b = _fit_line_from_two_points(xl1, yl1, xl2, yl2)

    if upper_a >= 0 or lower_a <= 0:
        return {"found": False, "reason": "Hai đường xu hướng chưa hội tụ đúng dạng tam giác cân.", "maxima": maxima_df, "minima": minima_df}
    if abs(upper_a - lower_a) < 1e-10:
        return {"found": False, "reason": "Hai trendline gần song song, không tạo giao điểm rõ ràng.", "maxima": maxima_df, "minima": minima_df}

    intersection_x = (lower_b - upper_b) / (upper_a - lower_a)
    intersection_y = upper_a * intersection_x + upper_b
    return {
        "found": True,
        "reason": "Tìm thấy cấu trúc Lower Highs và Higher Lows.",
        "upper_points": selected_highs,
        "lower_points": selected_lows,
        "upper_line": (upper_a, upper_b),
        "lower_line": (lower_a, lower_b),
        "intersection_x": float(intersection_x),
        "intersection_y": float(intersection_y),
        "maxima": maxima_df,
        "minima": minima_df,
    }


def build_advanced_backtest_figure(data: pd.DataFrame, result_dict: dict, pre_chart_days: int = 62, post_expiration_days: int = 10, triangle_order: int = 5) -> tuple[go.Figure, dict]:
    """Tạo chart nâng cao: nến + volume + S0/K + ngày định giá/đáo hạn + tam giác cân."""
    chart_start = result_dict["start_actual"] - pd.Timedelta(days=int(pre_chart_days))
    chart_end = result_dict["expiry_actual"] + pd.Timedelta(days=int(post_expiration_days))
    chart_data = data.loc[(data.index >= chart_start) & (data.index <= chart_end)].copy()
    if chart_data.empty:
        raise ValueError("Không có dữ liệu để vẽ biểu đồ.")
    if "Volume" not in chart_data.columns:
        chart_data["Volume"] = 0.0

    triangle = detect_symmetrical_triangle(data=data, valuation_date=result_dict["start_actual"], lookback_days=int(pre_chart_days), order=int(triangle_order))

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
        subplot_titles=("Biểu đồ nến, vùng định giá và mô hình kỹ thuật", "Khối lượng giao dịch"),
    )

    fig.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["Open"],
            high=chart_data["High"],
            low=chart_data["Low"],
            close=chart_data["Close"],
            name=f"{result_dict['symbol']} OHLC",
            increasing_line_color="#22C55E",
            decreasing_line_color="#EF4444",
            increasing_fillcolor="#22C55E",
            decreasing_fillcolor="#EF4444",
        ),
        row=1,
        col=1,
    )

    volume_colors = np.where(chart_data["Close"] >= chart_data["Open"], "rgba(34,197,94,0.55)", "rgba(239,68,68,0.55)")
    fig.add_trace(go.Bar(x=chart_data.index, y=chart_data["Volume"], name="Volume", marker_color=volume_colors, opacity=0.85), row=2, col=1)

    if triangle.get("found"):
        all_dates = pd.Series(np.arange(len(data)), index=data.index)
        upper_a, upper_b = triangle["upper_line"]
        lower_a, lower_b = triangle["lower_line"]
        x_start_idx = float(all_dates.loc[chart_data.index[0]])
        x_end_idx = float(all_dates.loc[chart_data.index[-1]])
        intersection_x = float(triangle["intersection_x"])
        x_line_end = min(max(intersection_x, x_start_idx), x_end_idx)
        x_line_values = np.linspace(x_start_idx, x_line_end, 120)
        x_dates = [data.index[int(np.clip(round(xv), 0, len(data) - 1))] for xv in x_line_values]
        fig.add_trace(go.Scatter(x=x_dates, y=upper_a * x_line_values + upper_b, mode="lines", line=dict(color="#3B82F6", width=3), name="Tam giác cân - Cản trên"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_dates, y=lower_a * x_line_values + lower_b, mode="lines", line=dict(color="#60A5FA", width=3), name="Tam giác cân - Hỗ trợ dưới"), row=1, col=1)
        upper_points = triangle["upper_points"]
        lower_points = triangle["lower_points"]
        fig.add_trace(go.Scatter(x=upper_points.index, y=upper_points["High"], mode="markers", marker=dict(size=10, color="#93C5FD", symbol="triangle-down"), name="Lower Highs dùng để fit"), row=1, col=1)
        fig.add_trace(go.Scatter(x=lower_points.index, y=lower_points["Low"], mode="markers", marker=dict(size=10, color="#BFDBFE", symbol="triangle-up"), name="Higher Lows dùng để fit"), row=1, col=1)

    fig.add_hline(y=float(result_dict["s0"]), line_width=2, line_dash="dash", line_color="#38BDF8", annotation_text=f"S0 = {float(result_dict['s0']):.2f}", annotation_position="bottom right", row=1, col=1)
    fig.add_hline(y=float(result_dict["k"]), line_width=3, line_dash="dash", line_color="#F59E0B", annotation_text=f"K mục tiêu = {float(result_dict['k']):.2f}", annotation_position="top right", row=1, col=1)

    y_max = float(chart_data["High"].max())
    y_min = float(chart_data["Low"].min())
    y_ann = y_max + max(y_max - y_min, 1.0) * 0.05
    fig.add_vline(x=result_dict["start_actual"], line_width=2, line_dash="dot", line_color="#FBBF24", row=1, col=1)
    fig.add_annotation(x=result_dict["start_actual"], y=y_ann, text=f"Ngày định giá<br>{result_dict['start_actual'].date()}", showarrow=True, arrowhead=2, ax=0, ay=-45, bgcolor="#111827", bordercolor="#FBBF24", font=dict(color="#F9FAFB", size=12), row=1, col=1)
    fig.add_vline(x=result_dict["expiry_actual"], line_width=2, line_dash="dot", line_color="#A78BFA", row=1, col=1)
    fig.add_annotation(x=result_dict["expiry_actual"], y=y_ann, text=f"Ngày đáo hạn<br>{result_dict['expiry_actual'].date()}", showarrow=True, arrowhead=2, ax=0, ay=-45, bgcolor="#111827", bordercolor="#A78BFA", font=dict(color="#F9FAFB", size=12), row=1, col=1)

    triangle_status = "Có nhận diện Tam giác cân" if triangle.get("found") else f"Không nhận diện được Tam giác cân: {triangle.get('reason', 'Không rõ lý do')}"
    fig.update_layout(
        title=(f"Backtest Call Option {result_dict['symbol']} | S0={float(result_dict['s0']):.2f}, K={float(result_dict['k']):.2f}, ST={float(result_dict['st_expiry']):.2f}<br><sup>{triangle_status}</sup>"),
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        height=820,
        margin=dict(l=35, r=35, t=90, b=35),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor="rgba(148,163,184,0.16)", row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, showgrid=True, gridcolor="rgba(148,163,184,0.16)", row=2, col=1)
    fig.update_yaxes(title_text="Giá cổ phiếu", showgrid=True, gridcolor="rgba(148,163,184,0.16)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", showgrid=True, gridcolor="rgba(148,163,184,0.12)", row=2, col=1)
    return fig, triangle


def format_advanced_backtest_report(result_dict: dict, triangle: dict) -> str:
    """Tạo báo cáo kiểm thử Black-Scholes có thêm trạng thái mô hình kỹ thuật."""
    base_report = format_backtest_report(result_dict)
    if triangle.get("found"):
        tech_line = "Mô hình kỹ thuật                         : Phát hiện cấu trúc Tam giác cân trước ngày định giá"
    else:
        tech_line = f"Mô hình kỹ thuật                         : Không phát hiện rõ Tam giác cân ({triangle.get('reason', 'Không rõ lý do')})"
    return base_report + "\n" + "-" * 78 + "\n" + tech_line


def monte_carlo_call_price_from_terminal(st_values: np.ndarray, strike: float, r: float, t_years: float) -> tuple[float, float]:
    """Discount average call payoff from simulated terminal prices."""
    payoff = np.maximum(st_values - strike, 0.0)
    discounted_price = np.exp(-r * t_years) * np.mean(payoff)
    return float(discounted_price), float(np.mean(payoff))


def monte_carlo_option_price_from_terminal(
    st_values: np.ndarray, strike: float, r: float, t_years: float, option_type: str
) -> tuple[float, float]:
    """Discount average payoff from simulated terminal prices (Call/Put)."""
    if option_type == "Put":
        payoff = np.maximum(strike - st_values, 0.0)
    else:
        payoff = np.maximum(st_values - strike, 0.0)
    discounted_price = np.exp(-r * t_years) * np.mean(payoff)
    return float(discounted_price), float(np.mean(payoff))


def binomial_tree_price(s: float, k: float, t_years: float, r: float, sigma: float, n: int, option_type: str) -> float:
    """CRR binomial tree pricing (European) to compare with BSM."""
    n_eff = max(1, int(n))
    dt = t_years / n_eff
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    p = min(max(float(p), 0.0), 1.0)
    disc = np.exp(-r * dt)

    stock = np.array([s * (u**j) * (d ** (n_eff - j)) for j in range(n_eff + 1)])
    if option_type == "Call":
        option_vals = np.maximum(stock - k, 0.0)
    else:
        option_vals = np.maximum(k - stock, 0.0)

    for step in range(n_eff - 1, -1, -1):
        option_vals = disc * (p * option_vals[1 : step + 2] + (1 - p) * option_vals[: step + 1])

    return float(option_vals[0])


def build_binomial_trees_ud(
    s0: float, k: float, t_years: float, r: float, u: float, d: float, n_steps: int, option_type: str, option_style: str
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Build stock and option trees using custom u/d with European or American exercise."""
    n_eff = max(1, int(n_steps))
    t_eff = max(float(t_years), 1e-10)
    dt = t_eff / n_eff
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)
    p = float(min(max(p, 0.0), 1.0))

    stock_tree = np.full((n_eff + 1, n_eff + 1), np.nan)
    option_tree = np.full((n_eff + 1, n_eff + 1), np.nan)

    for i in range(n_eff + 1):
        for j in range(i + 1):
            stock_tree[i, j] = s0 * (u**j) * (d ** (i - j))

    last_prices = stock_tree[n_eff, : n_eff + 1]
    if option_type == "Put":
        option_tree[n_eff, : n_eff + 1] = np.maximum(k - last_prices, 0.0)
    else:
        option_tree[n_eff, : n_eff + 1] = np.maximum(last_prices - k, 0.0)

    for i in range(n_eff - 1, -1, -1):
        for j in range(i + 1):
            continuation = disc * (p * option_tree[i + 1, j + 1] + (1 - p) * option_tree[i + 1, j])
            if option_type == "Put":
                intrinsic = max(k - stock_tree[i, j], 0.0)
            else:
                intrinsic = max(stock_tree[i, j] - k, 0.0)
            if option_style == "American":
                option_tree[i, j] = max(continuation, intrinsic)
            else:
                option_tree[i, j] = continuation

    return float(option_tree[0, 0]), stock_tree, option_tree, dt, p


def triangular_tree_to_dataframe(tree: np.ndarray, decimals: int = 4) -> pd.DataFrame:
    """Convert triangular tree to a clean DataFrame for display."""
    n = tree.shape[0] - 1
    rows = []
    for i in range(n + 1):
        row = {"Bước": i}
        for j in range(n + 1):
            row[f"Nút {j}"] = "" if j > i or np.isnan(tree[i, j]) else f"{tree[i, j]:.{decimals}f}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_lattice_graph_figure(
    stock_tree: np.ndarray, option_tree: np.ndarray, steps: int, k: float, r: float, dt: float, p: float
) -> go.Figure:
    """Build a detailed node-link lattice graph with edge probabilities."""
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_mid_x: list[float] = []
    edge_mid_y: list[float] = []
    edge_labels: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_color: list[str] = []

    for i in range(steps + 1):
        for j in range(i + 1):
            x0 = float(i)
            y0 = float(stock_tree[i, j])
            node_x.append(x0)
            node_y.append(y0)
            up_moves = j
            down_moves = i - j
            state_label = "S" + ("*u" * up_moves) + ("*d" * down_moves) if i > 0 else "S"
            node_text.append(f"{state_label}<br>S={stock_tree[i, j]:.2f}<br>v={option_tree[i, j]:.4f}")

            if i == 0:
                node_color.append("#22C55E")
            elif i == steps:
                node_color.append("#EF4444")
            else:
                node_color.append("#F59E0B")

            if i < steps:
                x1_up = float(i + 1)
                y1_up = float(stock_tree[i + 1, j + 1])
                edge_x.extend([x0, x1_up, None])
                edge_y.extend([y0, y1_up, None])
                edge_mid_x.append((x0 + x1_up) / 2.0)
                edge_mid_y.append((y0 + y1_up) / 2.0)
                edge_labels.append("p")

                x1_down = float(i + 1)
                y1_down = float(stock_tree[i + 1, j])
                edge_x.extend([x0, x1_down, None])
                edge_y.extend([y0, y1_down, None])
                edge_mid_x.append((x0 + x1_down) / 2.0)
                edge_mid_y.append((y0 + y1_down) / 2.0)
                edge_labels.append("1-p")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(148,163,184,0.45)", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=14, color=node_color, line=dict(color="#E5E7EB", width=1)),
            text=node_text,
            textposition="top center",
            textfont=dict(size=10, color="#E5E7EB"),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=edge_mid_x,
            y=edge_mid_y,
            mode="text",
            text=edge_labels,
            textfont=dict(size=12, color="#E5E7EB"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_annotation(
        x=0.5,
        y=1.06,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=f"Ghi chú: K={k:.2f}, r={r:.4f}, Δt={dt:.4f}, p={p:.4f}",
        font=dict(size=12, color="#E5E7EB"),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        xaxis_title="Bước thời gian (Step)",
        yaxis_title="Giá cổ phiếu",
        margin=dict(l=25, r=25, t=30, b=25),
        height=560,
    )
    fig.update_xaxes(dtick=1, showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return fig


apply_dark_theme()
st.markdown('<div class="hero">Bảng mô phỏng do Nhóm 5 thực hiện</div>', unsafe_allow_html=True)
st.title("Nhóm 5: Hội Những Người Cợt Nhả")
st.header("Mô hình định giá quyền chọn Black-Scholes-Merton (BSM)")

with st.sidebar:
    st.header("THAM SỐ MÔ HÌNH BSM")
    common_ticker = st.text_input("Mã cổ phiếu dùng chung cho 2 mô hình", value="ASML.AS")
    st.caption(f"Thị trường Châu Âu: dùng chung mã `{common_ticker}` cho Black-Scholes và Cây nhị thức.")
    asml_spot, asml_error = load_latest_close(common_ticker)
    if asml_error:
        st.warning(asml_error)
    s_default = float(asml_spot) if asml_spot is not None else 100.0
    s = st.number_input(f"S - Giá hiện tại ({common_ticker})", min_value=1.0, value=s_default, step=1.0)
    k = st.number_input("K - Giá thực thi", min_value=1.0, value=100.0, step=1.0)
    t_days = st.number_input("T - Thời gian đáo hạn (ngày)", min_value=1, value=30, step=1)
    r = st.number_input("r - Lãi suất phi rủi ro", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    sigma = st.number_input("sigma - Biến động (%)", min_value=1.0, max_value=300.0, value=20.0, step=1.0)
    option_type = st.selectbox("Loại quyền chọn", ["Call", "Put"])
    seed = st.number_input("Hạt giống ngẫu nhiên (seed) mô phỏng", min_value=0, value=42, step=1)

    with st.expander("Tùy chỉnh mô phỏng GBM (đường giá)", expanded=False):
        gbm_paths = st.slider("Số đường giá hiển thị", min_value=1, max_value=50, value=10, step=1, key="bsm_gbm_paths")
        gbm_steps = st.slider("Số bước thời gian", min_value=20, max_value=500, value=140, step=10, key="bsm_gbm_steps")

    with st.expander("Tùy chỉnh Cây nhị thức (CRR)", expanded=False):
        crr_s = st.number_input("S (CRR) - Giá cổ phiếu hiện tại", min_value=1.0, value=float(s), step=1.0, key="bsm_crr_s")
        crr_k = st.number_input("K (CRR) - Giá thực thi", min_value=1.0, value=float(k), step=1.0, key="bsm_crr_k")
        crr_t_days = st.number_input("T (CRR) - Số ngày đến đáo hạn", min_value=1, value=int(t_days), step=1, key="bsm_crr_t_days")
        crr_r = st.number_input("r (CRR) - Lãi suất phi rủi ro", min_value=0.0, max_value=1.0, value=float(r), step=0.01, key="bsm_crr_r")
        crr_sigma_pct = st.number_input(
            "sigma (CRR) - Biến động (%)", min_value=0.1, max_value=300.0, value=float(sigma), step=0.1, key="bsm_crr_sigma_pct"
        )
        crr_option_type = st.selectbox("Loại quyền chọn (CRR)", ["Call", "Put"], key="bsm_crr_option_type")
        crr_steps = st.slider("Số bước cây (n)", min_value=1, max_value=500, value=100, step=1, key="bsm_crr_steps")
    with st.expander("Mô hình Cây nhị thức theo thị trường (Mỹ/Châu Âu)", expanded=False):
        market_scenario = st.selectbox(
            "Kịch bản mẫu",
            [
                "Kịch bản 1 - Quyền chọn mua kiểu Mỹ",
                "Kịch bản 2 - Quyền chọn bán kiểu Châu Âu",
                "Tùy chỉnh thủ công",
            ],
            key="bsm_market_scenario",
        )
        if market_scenario == "Kịch bản 1 - Quyền chọn mua kiểu Mỹ":
            bt_symbol = common_ticker
            bt_s0 = st.number_input("S0", min_value=1.0, value=float(s_default), step=1.0, key="bt_s0_us")
            bt_k = st.number_input("K", min_value=1.0, value=float(s_default), step=1.0, key="bt_k_us")
            bt_t = st.number_input("T (năm)", min_value=0.01, value=1.0, step=0.1, key="bt_t_us")
            bt_r = st.number_input("r", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="bt_r_us")
            bt_u = st.number_input("u", min_value=0.01, value=1.1, step=0.01, key="bt_u_us")
            bt_d = st.number_input("d", min_value=0.01, value=0.9, step=0.01, key="bt_d_us")
            bt_n = st.slider("N (số bước)", min_value=1, max_value=20, value=5, step=1, key="bt_n_us")
            bt_option_type = "Call"
            bt_option_style = "American"
        elif market_scenario == "Kịch bản 2 - Quyền chọn bán kiểu Châu Âu":
            bt_symbol = common_ticker
            bt_s0 = st.number_input("S0", min_value=1.0, value=float(s_default), step=1.0, key="bt_s0_eu")
            bt_k = st.number_input("K", min_value=1.0, value=float(s_default * 1.05), step=1.0, key="bt_k_eu")
            bt_t = st.number_input("T (năm)", min_value=0.01, value=1.0, step=0.1, key="bt_t_eu")
            bt_r = st.number_input("r", min_value=0.0, max_value=1.0, value=0.03, step=0.01, key="bt_r_eu")
            bt_u = st.number_input("u", min_value=0.01, value=1.15, step=0.01, key="bt_u_eu")
            bt_d = st.number_input("d", min_value=0.01, value=0.85, step=0.01, key="bt_d_eu")
            bt_n = st.slider("N (số bước)", min_value=1, max_value=20, value=3, step=1, key="bt_n_eu")
            bt_option_type = "Put"
            bt_option_style = "European"
        else:
            bt_symbol = common_ticker
            bt_s0 = st.number_input("S0", min_value=1.0, value=float(s_default), step=1.0, key="bt_s0_custom")
            bt_k = st.number_input("K", min_value=1.0, value=float(s_default), step=1.0, key="bt_k_custom")
            bt_t = st.number_input("T (năm)", min_value=0.01, value=1.0, step=0.1, key="bt_t_custom")
            bt_r = st.number_input("r", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="bt_r_custom")
            bt_u = st.number_input("u", min_value=0.01, value=1.1, step=0.01, key="bt_u_custom")
            bt_d = st.number_input("d", min_value=0.01, value=0.9, step=0.01, key="bt_d_custom")
            bt_n = st.slider("N (số bước)", min_value=1, max_value=20, value=5, step=1, key="bt_n_custom")
            bt_option_type = st.selectbox("Loại quyền chọn", ["Call", "Put"], key="bt_option_type_custom")
            bt_option_style = st.selectbox("Kiểu quyền chọn", ["European", "American"], key="bt_option_style_custom")

    st.markdown("---")
    st.markdown("**Sản phẩm thực hiện bởi Nhóm 5**")

t_years = float(t_days) / 365.0
sigma_dec = float(sigma) / 100.0
result = black_scholes_calc(s, k, t_years, r, sigma_dec, option_type)
crr_t_years_top = float(crr_t_days) / 365.0
crr_sigma_dec_top = float(crr_sigma_pct) / 100.0
crr_price_top = binomial_tree_price(
    float(crr_s), float(crr_k), float(crr_t_years_top), float(crr_r), float(crr_sigma_dec_top), int(crr_steps), str(crr_option_type)
)

premium_col1, premium_col2 = st.columns(2)
with premium_col1:
    st.markdown(
        f"""
        <div class="premium-box">
            <div class="premium-label">Giá quyền chọn lý thuyết Black-Scholes ({option_type})</div>
            <div class="premium-value">${result["price"]:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with premium_col2:
    st.markdown(
        f"""
        <div class="premium-box">
            <div class="premium-label">Giá quyền chọn lý thuyết Cây nhị thức ({crr_option_type})</div>
            <div class="premium-value">${crr_price_top:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

model_tab_bs, model_tab_tree = st.tabs(["MÔ HÌNH BLACK-SCHOLES", "MÔ HÌNH CÂY NHỊ THỨC"])

with model_tab_bs:
    tab_formula, tab_market, tab_greeks = st.tabs(
        [
            "CÔNG THỨC BLACK-SCHOLES",
            "MÔ PHỎNG THỊ TRƯỜNG",
            "PHÂN TÍCH ĐỘ NHẠY",
        ]
    )

    with tab_formula:
        st.subheader("Công thức Black-Scholes cho quyền chọn kiểu Châu Âu")
        st.latex(r"d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}")
        st.latex(r"d_2 = d_1 - \sigma\sqrt{T}")
        st.latex(r"C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)")
        st.latex(r"P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("d1", f'{result["d1"]:.4f}')
        c2.metric("d2", f'{result["d2"]:.4f}')
        c3.metric("N(d1)", f'{result["Nd1"]:.4f}')
        c4.metric("N(d2)", f'{result["Nd2"]:.4f}')

        st.markdown(
            """
- `d1`: phản ánh mức độ thuận lợi của tài sản cơ sở sau khi điều chỉnh theo lãi suất và biến động.
- `d2`: phiên bản thận trọng hơn của `d1`, thể hiện xác suất trung hòa rủi ro tại ngày đáo hạn.
- `C`: giá lý thuyết của quyền chọn Mua (Call) kiểu Châu Âu.
- `P`: giá lý thuyết của quyền chọn Bán (Put) kiểu Châu Âu.
- `N(d1)` và `N(d2)`: các xác suất tích lũy từ phân phối chuẩn, là nền tảng cho việc định giá.
"""
        )
        st.info(
            "Mô hình Black-Scholes trong ứng dụng này tập trung vào quyền chọn kiểu Châu Âu, nghĩa là quyền chỉ được thực hiện tại thời điểm đáo hạn."
        )

    with tab_market:
        st.subheader("Mô phỏng đường giá bằng Geometric Brownian Motion")
        times, paths = simulate_gbm_paths(
            s, t_years, r, sigma_dec, n_steps=int(gbm_steps), n_paths=int(gbm_paths), seed=int(seed)
        )
        fig_market = go.Figure()
        for i in range(paths.shape[0]):
            fig_market.add_trace(
                go.Scatter(
                    x=times * 252,
                    y=paths[i],
                    mode="lines",
                    line=dict(width=1.5),
                    name=f"Path {i+1}",
                    opacity=0.75,
                )
            )
        fig_market.add_hline(y=k, line_dash="dash", line_color="#A3E635", annotation_text="Mức giá thực thi K")
        fig_market.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1220",
            plot_bgcolor="#0B1220",
            xaxis_title="Ngày tới đáo hạn",
            yaxis_title="Giá cổ phiếu",
            margin=dict(l=25, r=25, t=25, b=20),
            height=470,
        )
        st.plotly_chart(fig_market, width="stretch")

    with tab_greeks:
        st.subheader("Đường giá quyền chọn và các chỉ số Greek")
        s_axis = np.linspace(max(1.0, 0.5 * k), 1.5 * k, 120)
        prices_curve = []
        delta_curve = []
        gamma_curve = []
        for s_i in s_axis:
            res_i = black_scholes_calc(float(s_i), k, t_years, r, sigma_dec, option_type)
            prices_curve.append(res_i["price"])
            delta_curve.append(res_i["delta"])
            gamma_curve.append(res_i["gamma"])

        fig_price = go.Figure()
        fig_price.add_trace(
            go.Scatter(
                x=s_axis, y=prices_curve, mode="lines", line=dict(color="#A3E635", width=3), name="Giá quyền chọn"
            )
        )
        intrinsic = np.maximum(s_axis - k, 0) if option_type == "Call" else np.maximum(k - s_axis, 0)
        fig_price.add_trace(
            go.Scatter(
                x=s_axis,
                y=intrinsic,
                mode="lines",
                line=dict(color="#94A3B8", width=2, dash="dash"),
                name="Giá trị nội tại",
            )
        )
        fig_price.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0B1220",
            plot_bgcolor="#0B1220",
            xaxis_title="Giá cổ phiếu S",
            yaxis_title=f"Giá {option_type}",
            margin=dict(l=25, r=25, t=20, b=20),
            height=430,
        )
        st.plotly_chart(fig_price, width="stretch")

        g1, g2 = st.columns(2)
        g1.metric("Delta hiện tại", f'{result["delta"]:.5f}')
        g2.metric("Gamma hiện tại", f'{result["gamma"]:.5f}')

    st.markdown("---")
    with st.expander("Kiểm thử quá khứ Call Option ASML.AS", expanded=False):
        st.subheader("Backtest Call Option ASML.AS theo Black-Scholes")
        st.caption(
            "Module này nằm trong phần Black-Scholes: dùng dữ liệu quá khứ từ yfinance, historical volatility, "
            "lãi suất ECB €STR và K đề xuất theo Expected Move +1 độ lệch chuẩn. Biểu đồ nâng cao hiển thị nến, volume và tam giác cân."
        )

        with st.expander("Tham số backtest", expanded=False):
            b1, b2, b3 = st.columns(3)
            with b1:
                bt_symbol_bs = st.text_input("Mã cổ phiếu", value="ASML.AS", key="asml_backtest_symbol_inner").strip().upper()
                bt_start_date_bs = st.date_input("Ngày bắt đầu", value=pd.Timestamp("2023-11-01"), key="asml_backtest_start_inner")
            with b2:
                bt_tenor_days_bs = st.number_input("Kỳ hạn quyền chọn (ngày)", min_value=7, max_value=730, value=90, step=1, key="asml_backtest_tenor_inner")
                bt_lookback_bs = st.number_input("Lookback tính sigma (ngày giao dịch)", min_value=30, max_value=756, value=252, step=21, key="asml_backtest_lookback_inner")
            with b3:
                bt_std_mult_bs = st.number_input("Expected Move: số độ lệch chuẩn", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key="asml_backtest_std_inner")
                bt_rounding_bs = st.number_input("Làm tròn K theo bội số", min_value=1.0, max_value=50.0, value=5.0, step=1.0, key="asml_backtest_rounding_inner")
            bc1, bc2 = st.columns(2)
            with bc1:
                bt_fallback_rate_bs = st.number_input("Fallback r nếu ECB lỗi", min_value=-0.05, max_value=0.20, value=0.04, step=0.005, key="asml_backtest_fallback_inner")
            with bc2:
                bt_triangle_order = st.slider("Độ nhạy tìm đỉnh/đáy tam giác", min_value=2, max_value=15, value=5, step=1, key="asml_backtest_triangle_order_inner")

        try:
            bt_result, bt_price_data = build_asml_call_backtest(
                symbol=bt_symbol_bs,
                start_date=pd.Timestamp(bt_start_date_bs),
                tenor_days=int(bt_tenor_days_bs),
                lookback_days=int(bt_lookback_bs),
                std_mult=float(bt_std_mult_bs),
                rounding_base=float(bt_rounding_bs),
                fallback_rate=float(bt_fallback_rate_bs),
            )

            # 1) BIỂU ĐỒ NÂNG CAO HIỂN THỊ TRƯỚC
            # Biểu đồ gồm candlestick ở cửa sổ trên, volume ở cửa sổ dưới,
            # overlay S0/K, ngày định giá/ngày đáo hạn và trendline tam giác cân nếu nhận diện được.
            bt_fig, bt_triangle = build_advanced_backtest_figure(
                bt_price_data,
                bt_result,
                pre_chart_days=62,
                post_expiration_days=10,
                triangle_order=int(bt_triangle_order),
            )
            st.plotly_chart(bt_fig, width="stretch")

            # 2) BÁO CÁO TEXT HIỂN THỊ NGAY BÊN DƯỚI BIỂU ĐỒ
            st.markdown("### Báo cáo kiểm thử Black-Scholes")
            st.code(format_advanced_backtest_report(bt_result, bt_triangle), language="text")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Call Premium BSM", f"{float(bt_result['call_price']):.4f}")
            m2.metric("K đề xuất", f"{float(bt_result['k']):.2f}")
            m3.metric("Payoff đáo hạn", f"{float(bt_result['payoff']):.4f}")
            m4.metric("P/L", f"{float(bt_result['pnl']):.4f}", f"{float(bt_result['pnl_pct']):.2%}")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Sigma lịch sử", f"{float(bt_result['sigma_hist']):.2%}")
            m6.metric("r", f"{float(bt_result['r']):.2%}")
            m7.metric("d1", f"{float(bt_result['d1']):.4f}")
            m8.metric("d2", f"{float(bt_result['d2']):.4f}")

            if bt_triangle.get("found"):
                st.success("Đã nhận diện và vẽ mô hình Tam giác cân dựa trên Lower Highs và Higher Lows trước ngày định giá.")
            else:
                st.warning(f"Không vẽ được Tam giác cân rõ ràng: {bt_triangle.get('reason', 'Không rõ lý do')}")

        except Exception as exc:
            st.error(f"Không chạy được backtest ASML Call: {exc}")

with model_tab_tree:
    st.subheader("Mô hình Cây nhị thức (Binomial Tree)")

    crr_t_years = float(crr_t_days) / 365.0
    crr_sigma_dec = float(crr_sigma_pct) / 100.0
    crr_price = binomial_tree_price(
        float(crr_s), float(crr_k), float(crr_t_years), float(crr_r), float(crr_sigma_dec), int(crr_steps), str(crr_option_type)
    )
    crr_bsm_ref = black_scholes_calc(
        float(crr_s), float(crr_k), float(crr_t_years), float(crr_r), float(crr_sigma_dec), str(crr_option_type)
    )

    tree_tab_formula, tree_tab_lattice, tree_tab_pricing = st.tabs(
        ["LÝ THUYẾT CRR", "ĐỒ THỊ CÂY NHỊ THỨC", "ĐỊNH GIÁ & SO SÁNH"]
    )

    with tree_tab_formula:
        st.markdown("### Công thức Cây nhị thức Cox-Ross-Rubinstein (CRR)")
        st.latex(r"\Delta t = \frac{T}{n}")
        st.latex(r"u = e^{\sigma\sqrt{\Delta t}},\quad d = \frac{1}{u}")
        st.latex(r"p = \frac{e^{r\Delta t} - d}{u - d}")
        st.latex(r"V_{i,j} = e^{-r\Delta t}\left(pV_{i+1,j+1} + (1-p)V_{i+1,j}\right)")
        st.markdown(
            """
- `u`, `d`: hệ số tăng/giảm của giá cổ phiếu trong mỗi bước thời gian.
- `p`: xác suất trung hòa rủi ro dùng để chiết khấu kỳ vọng.
- Giá quyền chọn được tính bằng **quy nạp lùi** từ ngày đáo hạn về hiện tại.
- Với quyền chọn kiểu Mỹ: tại mỗi nút xét `max(giữ tiếp, thực hiện sớm)`.
"""
        )

    with tree_tab_lattice:
        st.subheader("Mô hình Cây nhị thức theo thị trường (u/d tùy chỉnh)")
        if bt_u <= bt_d:
            st.error("Điều kiện mô hình không hợp lệ: cần `u > d` để xây dựng cây nhị thức.")
        else:
            bt_price, bt_stock_tree, bt_option_tree, bt_dt, bt_p = build_binomial_trees_ud(
                float(bt_s0),
                float(bt_k),
                float(bt_t),
                float(bt_r),
                float(bt_u),
                float(bt_d),
                int(bt_n),
                str(bt_option_type),
                str(bt_option_style),
            )
            bt_m1, bt_m2, bt_m3, bt_m4 = st.columns(4)
            bt_m1.metric("Mã cổ phiếu mô phỏng", bt_symbol.upper())
            bt_style_vn = "Kiểu Mỹ" if bt_option_style == "American" else "Kiểu Châu Âu"
            bt_type_vn = "Quyền chọn mua (Call)" if bt_option_type == "Call" else "Quyền chọn bán (Put)"
            bt_m2.metric("Kiểu quyền chọn", f"{bt_style_vn} - {bt_type_vn}")
            bt_m3.metric("p trung hòa rủi ro", f"{bt_p:.4f}")
            bt_m4.metric("Giá quyền chọn hiện tại", f"{bt_price:.4f}")
            st.caption(
                f"Thiết lập cây: Δt = {bt_dt:.4f} năm, u = {bt_u:.4f}, d = {bt_d:.4f}, N = {int(bt_n)}."
            )

            st.markdown("### Đồ thị Lattice: mỗi nút hiển thị S và v (điều chỉnh theo công thức)")
            bt_graph_steps = int(bt_n)
            bt_graph_price, bt_graph_stock_tree, bt_graph_option_tree, bt_graph_dt, bt_graph_p = build_binomial_trees_ud(
                float(bt_s0),
                float(bt_k),
                float(bt_t),
                float(bt_r),
                float(bt_u),
                float(bt_d),
                int(bt_graph_steps),
                str(bt_option_type),
                str(bt_option_style),
            )
            lattice_fig = build_lattice_graph_figure(
                bt_graph_stock_tree,
                bt_graph_option_tree,
                int(bt_graph_steps),
                float(bt_k),
                float(bt_r),
                float(bt_graph_dt),
                float(bt_graph_p),
            )
            st.plotly_chart(lattice_fig, width="stretch")
            st.caption(
                f"Đồ thị được dựng theo tham số công thức hiện tại: n={bt_graph_steps}, "
                f"u={float(bt_u):.4f}, d={float(bt_d):.4f}, r={float(bt_r):.4f}, T={float(bt_t):.4f}."
            )

    with tree_tab_pricing:
        crr_c1, crr_c2, crr_c3 = st.columns(3)
        crr_c1.metric("Giá CRR", f"{crr_price:.4f}")
        crr_c2.metric("Số bước n", f"{int(crr_steps)}")
        crr_c3.metric("Chênh lệch so với BSM", f"{abs(crr_price - crr_bsm_ref['price']):.4f}")

        with st.expander("Xem đồ thị hội tụ theo số bước n", expanded=False):
            n_values = np.arange(10, 501, 10)
            bino_values = [
                binomial_tree_price(float(crr_s), float(crr_k), float(crr_t_years), float(crr_r), float(crr_sigma_dec), int(nv), str(crr_option_type))
                for nv in n_values
            ]
            bsm_level = np.full_like(n_values, crr_bsm_ref["price"], dtype=float)
            fig_conv = go.Figure()
            fig_conv.add_trace(
                go.Scatter(
                    x=n_values,
                    y=bsm_level,
                    mode="lines",
                    line=dict(color="#A3E635", width=2),
                    name="BSM (mốc chuẩn)",
                )
            )
            fig_conv.add_trace(
                go.Scatter(
                    x=n_values,
                    y=bino_values,
                    mode="lines+markers",
                    line=dict(color="#94A3B8", width=2, dash="dash"),
                    marker=dict(size=4),
                    name="CRR (cây nhị thức)",
                )
            )
            fig_conv.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0B1220",
                plot_bgcolor="#0B1220",
                xaxis_title="Số bước cây (n)",
                yaxis_title="Giá quyền chọn",
                margin=dict(l=25, r=25, t=20, b=20),
                height=420,
            )
            st.plotly_chart(fig_conv, width="stretch")

    if bt_u <= bt_d:
        st.error("Điều kiện mô hình không hợp lệ: cần `u > d` để xây dựng cây nhị thức.")
    else:
        bt_price, bt_stock_tree, bt_option_tree, bt_dt, bt_p = build_binomial_trees_ud(
            float(bt_s0),
            float(bt_k),
            float(bt_t),
            float(bt_r),
            float(bt_u),
            float(bt_d),
            int(bt_n),
            str(bt_option_type),
            str(bt_option_style),
        )
        if bt_option_style == "European":
            bt_sigma_equiv = max(np.sqrt(max(np.log(float(bt_u) / float(bt_d)), 0.0) / max(bt_dt, 1e-10)), 1e-10)
            bs_ref = black_scholes_calc(float(bt_s0), float(bt_k), float(bt_t), float(bt_r), float(bt_sigma_equiv), str(bt_option_type))
            bs_side = "call" if bt_option_type == "Call" else "put"
            st.info(
                f"So sánh tham chiếu với Black-Scholes ({bt_option_type}): "
                f"BS = {bs_ref[bs_side]:.4f}, Binomial = {bt_price:.4f}, sai số = {abs(bs_ref[bs_side]-bt_price):.4f}."
            )
        else:
            st.info(
                "Với quyền chọn kiểu Mỹ, mô hình cho phép thực hiện sớm tại mỗi nút: "
                "giá trị nút = max(giá trị tiếp tục, giá trị thực hiện ngay)."
            )


st.divider()
st.caption(
    "Hết phần Black-Scholes/GBM/CRR."
)
st.stop()
'''
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO
import streamlit as st
import yfinance as yf
import networkx as nx
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
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
                color: #FFFFFF !important;
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

tab_guide, tab_chain, tab_payoff = st.tabs(
    ["Lý thuyết chiến lược Butterfly",  "Bảng giá Quyền chọn", "Biểu đồ Thực chiến Bướm"]
)

with tab_guide:
    st.markdown(
        """
### Lý thuyết chiến lược Butterfly Spread

*1. Khái niệm:*
Butterfly Spread là chiến lược quyền chọn *trung lập về hướng giá* (market-neutral), được xây dựng bằng cách kết hợp *ba chân quyền chọn* cùng loại (Call hoặc Put) với ba mức strike khác nhau nhưng cách đều nhau. Chiến lược này phù hợp khi nhà đầu tư kỳ vọng giá tài sản cơ sở *ít biến động* và neo quanh một mức giá mục tiêu cho đến ngày đáo hạn.

*2. Cấu trúc chuẩn (Long Call Butterfly):*
- *Chân 1:* Mua 1 hợp đồng Call tại strike thấp (K1) — cánh dưới.
- *Chân 2:* Bán 2 hợp đồng Call tại strike giữa (K2) — tâm bướm.
- *Chân 3:* Mua 1 hợp đồng Call tại strike cao (K3) — cánh trên.
- Điều kiện cân bằng: *K2 − K1 = K3 − K2* (hai cánh cách đều tâm).

*3. Đặc điểm lợi nhuận (Payoff):*
- *Lãi tối đa:* đạt được khi giá đáo hạn = K2 (tâm bướm), bằng (K2 − K1) − phí ròng đã trả.
- *Lỗ tối đa:* giới hạn bằng phí ròng (net debit) đã trả khi mở vị thế.
- *Hai điểm hòa vốn:* BE1 = K1 + phí ròng, BE2 = K3 − phí ròng.

*4. Ưu điểm & Hạn chế:*
- Ưu điểm: Rủi ro giới hạn, chi phí thấp, tỷ lệ risk/reward hấp dẫn khi dự đoán đúng vùng giá.
- Hạn chế: Lợi nhuận tối đa chỉ đạt được trong biên độ hẹp quanh K2; nếu giá di chuyển mạnh ra ngoài K1–K3, vị thế sẽ lỗ toàn bộ phí.

*5. Khi nào nên dùng Butterfly?*
- Thị trường sideway, biến động ngụ ý (IV) được kỳ vọng *giảm* sau sự kiện (earnings, FOMC...).
- Có cơ sở kỹ thuật để dự đoán vùng "neo giá" (max pain, vùng hỗ trợ/kháng cự mạnh).
- Nhà đầu tư muốn giới hạn rủi ro với chi phí vốn thấp.
        """
    )

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
            width="stretch",
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
            width="stretch",
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
    st.plotly_chart(payoff_fig, width="stretch")

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
        width="stretch",
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
    st.plotly_chart(scenario_fig, width="stretch")

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
    st.plotly_chart(key_fig, width="stretch")


# =========================
# Brain nhóm: CRR Module
# =========================
def crr_intrinsic(stock_price: float, strike: float, option_type: str) -> float:
    if option_type == "Call":
        return max(stock_price - strike, 0.0)
    return max(strike - stock_price, 0.0)


def binomial_tree_pricing(
    s0: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    steps: int,
    option_type: str,
    option_style: str,
) -> tuple[float, np.ndarray, np.ndarray, float, float, float, float]:
    """CRR binomial pricing using backward induction."""
    dt = maturity / steps
    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    growth = float(np.exp(rate * dt))
    p = (growth - d) / (u - d)
    disc = float(np.exp(-rate * dt))

    stock_tree = np.full((steps + 1, steps + 1), np.nan)
    option_tree = np.full((steps + 1, steps + 1), np.nan)

    for i in range(steps + 1):
        for j in range(i + 1):
            stock_tree[i, j] = s0 * (u ** j) * (d ** (i - j))

    for j in range(steps + 1):
        option_tree[steps, j] = crr_intrinsic(stock_tree[steps, j], strike, option_type)

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            continuation = disc * (p * option_tree[i + 1, j + 1] + (1 - p) * option_tree[i + 1, j])
            if option_style == "American":
                exercise = crr_intrinsic(stock_tree[i, j], strike, option_type)
                option_tree[i, j] = max(continuation, exercise)
            else:
                option_tree[i, j] = continuation

    return float(option_tree[0, 0]), stock_tree, option_tree, dt, u, d, p


def build_crr_tree_figure(stock_tree: np.ndarray, option_tree: np.ndarray, steps: int) -> go.Figure:
    """Visualize binomial stock/option lattice with labels."""
    graph = nx.DiGraph()
    x_edges, y_edges = [], []
    node_x, node_y, node_text, node_color = [], [], [], []

    for i in range(steps + 1):
        for j in range(i + 1):
            graph.add_node((i, j))
            x = float(i)
            y = float(stock_tree[i, j])
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"S={stock_tree[i, j]:.2f}<br>v={option_tree[i, j]:.2f}")
            if i == 0:
                node_color.append("#22C55E")
            elif i == steps:
                node_color.append("#EF4444")
            else:
                node_color.append("#F59E0B")

            if i < steps:
                for nxt in [(i + 1, j), (i + 1, j + 1)]:
                    graph.add_edge((i, j), nxt)
                    x_edges.extend([x, float(nxt[0]), None])
                    y_edges.extend([y, float(stock_tree[nxt[0], nxt[1]]), None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_edges,
            y=y_edges,
            mode="lines",
            line=dict(color="rgba(148,163,184,0.5)", width=1.5),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            marker=dict(size=22, color=node_color, line=dict(color="#E5E7EB", width=1)),
            text=node_text,
            textposition="top center",
            textfont=dict(size=10, color="#E5E7EB"),
            hovertemplate="%{text}<extra></extra>",
            name="Nodes",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        xaxis_title="Bước thời gian (Step)",
        yaxis_title="Giá cổ phiếu",
        margin=dict(l=25, r=25, t=30, b=25),
        showlegend=False,
        height=560,
    )
    fig.update_xaxes(dtick=1, showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)")
    return fig


st.divider()
st.caption("Chế độ học thuật: mô phỏng thị trường và định giá quyền chọn bằng backward induction.")

with st.sidebar:
    st.markdown("---")
    st.subheader("Tham số CRR")
    crr_s = st.number_input("S - Giá cổ phiếu hiện tại", min_value=1.0, value=100.0, step=1.0, key="crr_s")
    crr_k = st.number_input("K - Giá thực thi", min_value=1.0, value=100.0, step=1.0, key="crr_k")
    crr_t = st.number_input("T - Thời gian đáo hạn (năm)", min_value=0.01, value=1.0, step=0.1, key="crr_t")
    crr_r = st.number_input("r - Lãi suất phi rủi ro", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="crr_r")
    crr_sigma = st.number_input("sigma - Độ biến động", min_value=0.01, max_value=2.0, value=0.2, step=0.01, key="crr_sigma")
    crr_n = st.slider("n - Số bước cây", min_value=1, max_value=5, value=3, step=1, key="crr_n")
    crr_type = st.selectbox("Loại quyền chọn", ["Call", "Put"], key="crr_type")
    crr_style = st.selectbox("Kiểu quyền chọn", ["European", "American"], key="crr_style")
    crr_seed = st.number_input("Seed mô phỏng thị trường", min_value=0, value=42, step=1, key="crr_seed")

try:
    crr_price, crr_stock_tree, crr_opt_tree, crr_dt, crr_u, crr_d, crr_p = binomial_tree_pricing(
        crr_s, crr_k, crr_t, crr_r, crr_sigma, int(crr_n), crr_type, crr_style
    )
except Exception as exc:
    st.error(f"Lỗi tính toán CRR: {exc}")
    st.stop()

crr_tab1, crr_tab2, crr_tab3, crr_tab4 = st.tabs(
    ["Lý thuyết CRR", "Mô phỏng Thị trường", "Trực quan hóa Cây Nhị thức", "Động cơ Định giá"]
)

with crr_tab1:
    st.subheader("Công thức nền tảng CRR")
    st.latex(r"\Delta t = \frac{T}{n}")
    st.latex(r"u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}")
    st.latex(r"p = \frac{e^{r\Delta t} - d}{u - d}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delta t", f"{crr_dt:.4f}")
    c2.metric("u (Bước tăng)", f"{crr_u:.4f}")
    c3.metric("d (Bước giảm)", f"{crr_d:.4f}")
    c4.metric("p (Xác suất tăng)", f"{crr_p:.4f}")
    if not (0 <= crr_p <= 1):
        st.warning("Xác suất trung hòa rủi ro p nằm ngoài [0,1]. Hãy điều chỉnh n/r/sigma để mô hình hợp lệ hơn.")

with crr_tab2:
    st.subheader("Random Walk / GBM minh họa chuyển động thị trường")
    np.random.seed(int(crr_seed))
    points = 160
    dt_sim = crr_t / points
    shocks = np.random.normal(0, np.sqrt(dt_sim), points)
    prices = [crr_s]
    for z in shocks:
        next_price = prices[-1] * np.exp((crr_r - 0.5 * crr_sigma**2) * dt_sim + crr_sigma * z)
        prices.append(float(next_price))
    t_axis = np.linspace(0, crr_t, len(prices))
    sim_fig = go.Figure()
    sim_fig.add_trace(
        go.Scatter(
            x=t_axis,
            y=prices,
            mode="lines",
            line=dict(color="#22D3EE", width=2),
            name="Giá mô phỏng",
        )
    )
    sim_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0B1220",
        xaxis_title="Thời gian (năm)",
        yaxis_title="Giá cổ phiếu",
        margin=dict(l=25, r=25, t=25, b=20),
        height=420,
    )
    st.plotly_chart(sim_fig, width="stretch")

with crr_tab3:
    st.subheader("Lattice Graph: mỗi node hiển thị S và v")
    tree_fig = build_crr_tree_figure(crr_stock_tree, crr_opt_tree, int(crr_n))
    st.plotly_chart(tree_fig, width="stretch")

with crr_tab4:
    st.subheader("Hàm định giá bằng Backward Induction")
    st.code(
        """def binomial_tree_pricing(S, K, T, r, sigma, n, option_type, option_style):
    dt = T / n
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp(r * dt) - d) / (u - d)
    disc = exp(-r * dt)
    # 1) payoff tại đáo hạn
    # 2) lùi từng bước: v = disc * (p * v_up + (1-p) * v_down)
    # 3) nếu American: v = max(v, intrinsic)
    return v0""",
        language="python",
    )
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Giá trị quyền chọn lý thuyết hôm nay</div>
            <div class="metric-value">{crr_price:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
'''