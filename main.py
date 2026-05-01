"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MÔ PHỎNG ĐỊNH GIÁ QUYỀN CHỌN MUA (CALL OPTION) - MÔ HÌNH BLACK-SCHOLES  ║
║   Mã cổ phiếu : ASML Holding N.V. (ASML.AS - Sàn Euronext Amsterdam)       ║
║   Ngày bắt đầu: 01/11/2023  |  Kỳ hạn: ~90 ngày                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Luồng thực thi:
  1. Tải dữ liệu giá lịch sử ASML.AS từ yfinance
  2. Tính Biến động Lịch sử (Historical Volatility - σ) từ log-return
  3. Lấy lãi suất phi rủi ro (r) từ API ECB (lãi suất tái cấp vốn tại thời điểm lịch sử)
  4. Tự động đề xuất Giá thực hiện (Strike Price - K) = S₀ × exp(σ√T)  ← +1 Độ lệch chuẩn
  5. Định giá Call Option theo công thức Black-Scholes-Merton
  6. Vẽ biểu đồ nến (Candlestick) tương tác với Plotly (HIỂN THỊ TRƯỚC)
  7. In "Báo cáo kiểm thử" kết quả P&L ra console (SAU BIỂU ĐỒ)
"""

# ─────────────────────────── IMPORT THƯ VIỆN ───────────────────────────────
import sys
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")  # Bỏ qua các warning không cần thiết

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 1 – THAM SỐ MÔ PHỎNG (THAY ĐỔI TẠI ĐÂY ĐỂ THỬ CÁC KỊCH BẢN KHÁC)
# ══════════════════════════════════════════════════════════════════════════════

TICKER       = "ASML.AS"          # Mã chứng khoán trên Euronext Amsterdam
START_DATE   = date(2023, 11, 1)  # Ngày bắt đầu mô phỏng (ngày mua quyền chọn)
MATURITY_DAYS = 90                # Kỳ hạn hợp đồng (số ngày lịch)
HV_WINDOW    = 30                 # Số ngày giao dịch dùng để tính Historical Volatility
TRADING_DAYS = 252                # Số ngày giao dịch quy ước trong 1 năm
N_CONTRACTS  = 1                  # Số hợp đồng (1 hợp đồng = 100 cổ phiếu trên thị trường EU)
LOT_SIZE     = 100                # Số cổ phiếu mỗi hợp đồng

# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 2 – HÀM TIỆN ÍCH
# ══════════════════════════════════════════════════════════════════════════════

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Tính giá Call Option theo công thức Black-Scholes-Merton (BSM).

    Công thức:
        d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
        d₂ = d₁ - σ·√T
        C  = S·N(d₁) - K·e^(-rT)·N(d₂)

    Tham số:
        S     : Giá cổ phiếu hiện tại (spot price)
        K     : Giá thực hiện (strike price)
        T     : Thời gian đến đáo hạn (tính bằng NĂM)
        r     : Lãi suất phi rủi ro (dạng thập phân, ví dụ 4.5% → 0.045)
        sigma : Biến động (annualized volatility, dạng thập phân)

    Trả về: dict chứa giá Call, d1, d2, Delta, Gamma, Theta, Vega, Rho
    """
    T_eff     = max(T, 1e-10)
    sigma_eff = max(sigma, 1e-10)

    d1 = (np.log(S / K) + (r + 0.5 * sigma_eff ** 2) * T_eff) / (sigma_eff * np.sqrt(T_eff))
    d2 = d1 - sigma_eff * np.sqrt(T_eff)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T_eff) * norm.cdf(d2)

    # ── Greeks ──────────────────────────────────────────────────────────────
    delta = norm.cdf(d1)                                                  # Δ
    gamma = norm.pdf(d1) / (S * sigma_eff * np.sqrt(T_eff))              # Γ
    theta = (                                                              # Θ (theo ngày)
        -(S * norm.pdf(d1) * sigma_eff) / (2 * np.sqrt(T_eff))
        - r * K * np.exp(-r * T_eff) * norm.cdf(d2)
    ) / TRADING_DAYS
    vega  = S * norm.pdf(d1) * np.sqrt(T_eff) / 100                      # ν (per 1% σ)
    rho   = K * T_eff * np.exp(-r * T_eff) * norm.cdf(d2) / 100          # ρ (per 1% r)

    return {
        "call_price": float(call_price),
        "d1": float(d1), "d2": float(d2),
        "delta": float(delta), "gamma": float(gamma),
        "theta": float(theta), "vega": float(vega), "rho": float(rho),
    }


def calc_historical_volatility(prices: pd.Series, window: int) -> float:
    """
    Tính Biến động Lịch sử (Historical Volatility – σ) theo phương pháp log-return.

    σ_hàng_ngày = std(ln(Pₜ / Pₜ₋₁))  trong cửa sổ `window` ngày giao dịch
    σ_năm       = σ_hàng_ngày × √252

    Tham số:
        prices : pd.Series giá đóng cửa (đã điều chỉnh)
        window : số ngày giao dịch dùng để tính
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < window:
        # Nếu không đủ dữ liệu → dùng toàn bộ lịch sử
        sigma_daily = log_returns.std()
    else:
        sigma_daily = log_returns.iloc[-window:].std()
    return float(sigma_daily * np.sqrt(TRADING_DAYS))


def fetch_ecb_rate_at_date(target_date: date) -> float:
    """
    Lấy lãi suất tái cấp vốn chủ chốt (Main Refinancing Rate) của ECB
    tại thời điểm lịch sử gần nhất với `target_date` qua API ECB Data Portal.

    API endpoint: https://data-api.ecb.europa.eu/service/data/
    Series key  : FM/B.U2.EUR.4F.KR.MRR_FR.LEV  (Main Refinancing Rate – Fixed)

    Nếu gọi API thất bại → dùng giá trị dự phòng (fallback) = 4.5% (Nov 2023)
    """
    FALLBACK_RATE = 0.045  # Lãi suất ECB tháng 11/2023 ≈ 4.5%

    # Xây dựng URL truy vấn dữ liệu tháng ECB
    start_str = (target_date - timedelta(days=180)).strftime("%Y-%m")
    end_str   = target_date.strftime("%Y-%m")
    url = (
        "https://data-api.ecb.europa.eu/service/data/"
        "FM/B.U2.EUR.4F.KR.MRR_FR.LEV"
        f"?startPeriod={start_str}&endPeriod={end_str}"
        "&format=csvdata"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        lines = [ln for ln in resp.text.splitlines() if ln.strip()]

        # Lọc các dòng dữ liệu (bỏ header)
        data_lines = [ln for ln in lines if not ln.startswith("KEY") and "," in ln]
        if not data_lines:
            print(f"  [ECB API] Không có dữ liệu → dùng dự phòng {FALLBACK_RATE*100:.2f}%")
            return FALLBACK_RATE

        # Lấy dòng cuối cùng (gần nhất với target_date) → cột OBS_VALUE
        last_line = data_lines[-1]
        cols = last_line.split(",")

        # ECB CSV: cột OBS_VALUE thường ở vị trí index 7 hoặc cột có tên OBS_VALUE
        header_line = next((ln for ln in lines if ln.startswith("KEY")), None)
        if header_line:
            headers = header_line.split(",")
            try:
                obs_idx = headers.index("OBS_VALUE")
                rate_pct = float(cols[obs_idx])
            except (ValueError, IndexError):
                rate_pct = float(cols[-1])  # Fallback: lấy cột cuối
        else:
            rate_pct = float(cols[-1])

        rate = rate_pct / 100.0  # Chuyển từ % sang thập phân
        print(f"  [ECB API] Lãi suất tái cấp vốn ECB tại {target_date}: {rate_pct:.2f}%")
        return rate

    except Exception as exc:
        print(f"  [ECB API] Lỗi: {exc} → dùng dự phòng {FALLBACK_RATE*100:.2f}%")
        return FALLBACK_RATE


def fetch_price_data(ticker: str, from_date: date, to_date: date) -> pd.DataFrame:
    """
    Tải dữ liệu OHLCV (Open/High/Low/Close/Volume) từ yfinance.

    Tham số:
        ticker   : Mã chứng khoán (ví dụ: "ASML.AS")
        from_date: Ngày bắt đầu tải dữ liệu
        to_date  : Ngày kết thúc tải dữ liệu

    Trả về: DataFrame với cột [Open, High, Low, Close, Volume]
    """
    print(f"  [yfinance] Tải dữ liệu {ticker}: {from_date} → {to_date} ...")
    # Lấy thêm 60 ngày trước start_date để tính HV đủ window
    fetch_start = (from_date - timedelta(days=90)).strftime("%Y-%m-%d")
    fetch_end   = to_date.strftime("%Y-%m-%d")

    df = yf.download(
        ticker,
        start=fetch_start,
        end=fetch_end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        sys.exit(f"[LỖI] Không tải được dữ liệu {ticker}. Kiểm tra kết nối internet hoặc mã ticker.")

    # Chuẩn hóa cột nếu MultiIndex (do yfinance mới trả về MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).date  # Chuyển index sang date object
    print(f"  [yfinance] Tải thành công {len(df)} phiên giao dịch.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 3 – HÀM VẼ BIỂU ĐỒ NẾN (CANDLESTICK CHART) VỚI PLOTLY
# ══════════════════════════════════════════════════════════════════════════════

def build_candlestick_chart(
    df_chart: pd.DataFrame,
    ticker: str,
    start_date: date,
    expiration_date: date,
    S0: float,
    K: float,
    call_premium: float,
) -> go.Figure:
    """
    Vẽ biểu đồ nến (Candlestick Chart) tương tác với Plotly.

    Phạm vi hiển thị:
        Từ start_date  →  expiration_date + 10 ngày (để thấy xu hướng sau đáo hạn)

    Overlays (lớp phủ):
        ─ Đường ngang xanh dương : Giá mua ban đầu S₀
        ─ Đường đứt nét xanh lá  : Giá thực hiện K (Strike Price)
        ─ Vùng tô mờ             : Khoảng [S₀, K] – vùng lợi nhuận kỳ vọng

    Markers (chú thích):
        ● Ngày bắt đầu mô phỏng
        ● Ngày đáo hạn hợp đồng
    """
    # ── Lọc dữ liệu theo phạm vi hiển thị ────────────────────────────────────
    view_end = expiration_date + timedelta(days=10)  # Kéo dài 10 ngày sau đáo hạn
    mask = [start_date <= d <= view_end for d in df_chart.index]
    df_view = df_chart[mask].copy()

    if df_view.empty:
        print("[CẢNH BÁO] Không có dữ liệu để vẽ biểu đồ trong phạm vi đã chọn.")
        return go.Figure()

    dates       = [str(d) for d in df_view.index]
    close_vals  = df_view["Close"].values.flatten()

    # ── Giá đóng cửa tại ngày bắt đầu và đáo hạn ────────────────────────────
    start_str  = str(start_date)
    expire_str = str(expiration_date)

    # Tìm ngày giao dịch gần nhất với ngày đáo hạn (có thể là ngày nghỉ lễ)
    available_dates = list(df_view.index)
    expire_actual = min(available_dates, key=lambda d: abs((d - expiration_date).days))
    S_expire = float(df_view.loc[expire_actual, "Close"])

    # ── Khởi tạo Figure ───────────────────────────────────────────────────────
    fig = go.Figure()

    # 1) Biểu đồ nến OHLC chính
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df_view["Open"].values.flatten(),
        high=df_view["High"].values.flatten(),
        low=df_view["Low"].values.flatten(),
        close=close_vals,
        name=ticker,
        increasing_line_color="#22C55E",   # Nến tăng: xanh lá
        decreasing_line_color="#EF4444",   # Nến giảm: đỏ
        increasing_fillcolor="#22C55E",
        decreasing_fillcolor="#EF4444",
        whiskerwidth=0.3,
        line_width=1.2,
    ))

    # 2) Đường ngang S₀ – Giá mua ban đầu (màu xanh dương, nét liền)
    fig.add_hline(
        y=S0,
        line=dict(color="#38BDF8", width=1.8, dash="solid"),
        annotation_text=f"  S₀ = €{S0:.2f}  (Giá ban đầu)",
        annotation_position="top right",
        annotation_font=dict(color="#38BDF8", size=12),
    )

    # 3) Đường ngang K – Giá thực hiện (màu xanh lá, nét đứt nổi bật)
    fig.add_hline(
        y=K,
        line=dict(color="#A3E635", width=2.2, dash="dash"),
        annotation_text=f"  K = €{K:.2f}  (Strike Price – Mục tiêu +1σ)",
        annotation_position="bottom right",
        annotation_font=dict(color="#A3E635", size=12),
    )

    # 4) Vùng tô bóng giữa S₀ và K (vùng lợi nhuận kỳ vọng)
    fig.add_hrect(
        y0=S0, y1=K,
        fillcolor="rgba(163,230,53,0.06)",   # Xanh lá mờ
        line_width=0,
        annotation_text="Vùng lợi nhuận kỳ vọng",
        annotation_position="top left",
        annotation_font=dict(color="#A3E635", size=10),
    )

    # 5) Marker – Ngày bắt đầu mô phỏng (trục X)
    if start_str in dates:
        start_close = float(df_view.loc[start_date, "Close"])
        fig.add_trace(go.Scatter(
            x=[start_str],
            y=[start_close],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="#FACC15",
                        line=dict(color="#1E293B", width=1.5)),
            text=["▶ Bắt đầu"],
            textposition="top right",
            textfont=dict(color="#FACC15", size=11),
            name="Ngày bắt đầu",
            showlegend=True,
        ))

    # 6) Marker – Ngày đáo hạn hợp đồng
    fig.add_trace(go.Scatter(
        x=[str(expire_actual)],
        y=[S_expire],
        mode="markers+text",
        marker=dict(symbol="diamond", size=14, color="#F97316",
                    line=dict(color="#1E293B", width=1.5)),
        text=["◆ Đáo hạn"],
        textposition="top right",
        textfont=dict(color="#F97316", size=11),
        name=f"Đáo hạn ({expire_actual})",
        showlegend=True,
    ))

    # 7) Đường thẳng dọc tại ngày bắt đầu (dạng shape)
    if start_str in dates:
        fig.add_vline(
            x=start_str,
            line=dict(color="#FACC15", width=1.2, dash="dot"),
            annotation_text=" Ngày bắt đầu",
            annotation_position="top left",
            annotation_font=dict(color="#FACC15", size=10),
        )

    # 8) Đường thẳng dọc tại ngày đáo hạn
    fig.add_vline(
        x=str(expire_actual),
        line=dict(color="#F97316", width=1.2, dash="dot"),
        annotation_text=" Đáo hạn",
        annotation_position="top right",
        annotation_font=dict(color="#F97316", size=10),
    )

    # ── Annotation tổng hợp: hộp thông tin góc trên trái ────────────────────
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.97,
        showarrow=False,
        align="left",
        bgcolor="rgba(15,23,42,0.85)",
        bordercolor="#334155",
        borderwidth=1,
        borderpad=8,
        font=dict(color="#E2E8F0", size=11, family="Courier New"),
        text=(
            f"<b>MÔ PHỎNG BLACK-SCHOLES | {ticker}</b><br>"
            f"──────────────────────────<br>"
            f"Ngày bắt đầu : {start_date}<br>"
            f"Ngày đáo hạn : {expiration_date} ({MATURITY_DAYS}d)<br>"
            f"S₀ (Spot)    : €{S0:,.2f}<br>"
            f"K  (Strike)  : €{K:,.2f}<br>"
            f"Premium      : €{call_premium:.4f}/cổ phiếu<br>"
            f"S tại đáo hạn: €{S_expire:,.2f}"
        ),
    )

    # ── Định dạng layout tổng thể ────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{ticker} – Biểu đồ Nến + Mô phỏng Call Option BSM</b><br>"
                f"<sup>Giai đoạn: {start_date} → {view_end} | "
                f"Kỳ hạn: {MATURITY_DAYS} ngày | "
                f"K = €{K:.2f} (+1σ Expected Move)</sup>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=15, color="#F1F5F9"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0B1220",
        plot_bgcolor="#0D1526",
        font=dict(color="#CBD5E1", family="Inter, Arial"),
        xaxis=dict(
            title="Ngày giao dịch",
            rangeslider_visible=False,       # Tắt thanh kéo phía dưới
            showgrid=True,
            gridcolor="rgba(148,163,184,0.12)",
            tickangle=-30,
            tickfont=dict(size=10),
            type="category",                 # Bỏ khoảng trống cuối tuần
        ),
        yaxis=dict(
            title="Giá (EUR €)",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.12)",
            tickprefix="€",
            tickformat=",.0f",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right",  x=1,
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        margin=dict(l=60, r=60, t=100, b=60),
        height=620,
        hovermode="x unified",
    )

    # Tô màu nền weekday/weekend (không cần vì dùng type='category')
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor="#475569", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikecolor="#475569", spikethickness=1)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# PHẦN 4 – LUỒNG CHÍNH (MAIN EXECUTION FLOW)
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("  MÔ PHỎNG ĐỊNH GIÁ QUYỀN CHỌN MUA – BLACK-SCHOLES-MERTON")
    print(f"  Mã cổ phiếu: {TICKER}  |  Ngày bắt đầu: {START_DATE}")
    print("=" * 70)

    # ── Bước 1: Tính ngày đáo hạn ─────────────────────────────────────────────
    expiration_date = START_DATE + timedelta(days=MATURITY_DAYS)
    T_years = MATURITY_DAYS / 365.0       # Kỳ hạn tính theo NĂM (365 ngày lịch)
    T_trading = MATURITY_DAYS * (TRADING_DAYS / 365)  # Tương đương ngày giao dịch

    print(f"\n[BƯỚC 1] Thông tin kỳ hạn:")
    print(f"  Ngày bắt đầu  : {START_DATE}")
    print(f"  Ngày đáo hạn  : {expiration_date}  ({MATURITY_DAYS} ngày lịch)")
    print(f"  T (năm)       : {T_years:.4f}")

    # ── Bước 2: Tải dữ liệu giá từ yfinance ──────────────────────────────────
    print(f"\n[BƯỚC 2] Tải dữ liệu giá từ yfinance:")
    # Lấy đến expiration + 15 ngày để hiển thị đủ biểu đồ
    fetch_to = expiration_date + timedelta(days=15)
    df_all = fetch_price_data(TICKER, START_DATE, fetch_to)

    # Trích xuất giá đóng cửa tại ngày bắt đầu
    available = sorted(df_all.index)
    # Tìm ngày giao dịch đầu tiên >= START_DATE
    actual_start = next((d for d in available if d >= START_DATE), available[0])
    S0 = float(df_all.loc[actual_start, "Close"])

    print(f"  Ngày giao dịch thực tế đầu tiên: {actual_start}")
    print(f"  Giá đóng cửa S₀ = €{S0:,.4f}")

    # ── Bước 3: Tính Historical Volatility σ ──────────────────────────────────
    print(f"\n[BƯỚC 3] Tính Historical Volatility (σ) – cửa sổ {HV_WINDOW} ngày:")
    # Lấy dữ liệu Close TRƯỚC ngày bắt đầu để tính HV (không nhìn trước)
    pre_mask   = [d < actual_start for d in df_all.index]
    df_pre     = df_all[pre_mask]
    close_pre  = df_pre["Close"].squeeze()

    if len(close_pre) < 2:
        print("  [CẢNH BÁO] Không đủ dữ liệu lịch sử → dùng σ = 30%")
        sigma = 0.30
    else:
        sigma = calc_historical_volatility(close_pre, HV_WINDOW)

    sigma_daily = sigma / np.sqrt(TRADING_DAYS)
    print(f"  σ hàng ngày   : {sigma_daily*100:.3f}%")
    print(f"  σ hóa năm     : {sigma*100:.2f}%")

    # ── Bước 4: Lấy lãi suất phi rủi ro r từ ECB ─────────────────────────────
    print(f"\n[BƯỚC 4] Lấy lãi suất ECB tại ngày {START_DATE}:")
    r = fetch_ecb_rate_at_date(START_DATE)
    print(f"  r (lãi suất phi rủi ro) = {r*100:.2f}%")

    # ── Bước 5: Đề xuất Strike Price K (Expected Move +1σ) ───────────────────
    # Công thức: K = S₀ × exp(σ × √T)
    # Ý nghĩa  : Đây là mức giá mà thị trường kỳ vọng cổ phiếu CÓ XÁC SUẤT ~16%
    #             vượt qua trong kỳ hạn T (tương ứng với +1 Độ lệch chuẩn)
    print(f"\n[BƯỚC 5] Tự động đề xuất Strike Price K:")
    K = S0 * np.exp(sigma * np.sqrt(T_years))
    expected_move = K - S0
    expected_move_pct = (expected_move / S0) * 100

    print(f"  Công thức   : K = S₀ × exp(σ × √T)")
    print(f"             = €{S0:.2f} × exp({sigma:.4f} × √{T_years:.4f})")
    print(f"  K           = €{K:,.4f}")
    print(f"  Expected Move = +€{expected_move:.2f}  (+{expected_move_pct:.2f}% so với S₀)")
    print(f"  (Xác suất giá vượt K trong {MATURITY_DAYS} ngày ≈ 16% theo phân phối chuẩn)")

    # ── Bước 6: Định giá Call Option theo Black-Scholes ───────────────────────
    print(f"\n[BƯỚC 6] Định giá Call Option theo Black-Scholes-Merton:")
    bsm = black_scholes_call(S=S0, K=K, T=T_years, r=r, sigma=sigma)
    call_premium   = bsm["call_price"]
    total_cost     = call_premium * LOT_SIZE * N_CONTRACTS  # Tổng phí mua quyền chọn

    print(f"  Giá Call (BSM) = €{call_premium:.4f} / cổ phiếu")
    print(f"  Tổng phí quyền chọn ({N_CONTRACTS} hợp đồng × {LOT_SIZE} cổ phiếu)")
    print(f"               = €{total_cost:,.2f}")

    # ── Bước 7: Lấy giá thực tế tại ngày đáo hạn ─────────────────────────────
    print(f"\n[BƯỚC 7] Lấy giá thực tế tại ngày đáo hạn ({expiration_date}):")
    after_mask    = [d >= expiration_date for d in df_all.index]
    df_after      = df_all[after_mask]

    if df_after.empty:
        print("  [CẢNH BÁO] Chưa có dữ liệu tại ngày đáo hạn (có thể là ngày trong tương lai).")
        S_T = None
    else:
        expire_actual = sorted(df_after.index)[0]  # Ngày giao dịch gần nhất sau đáo hạn
        S_T = float(df_after.iloc[0]["Close"])
        print(f"  Ngày giao dịch thực tế: {expire_actual}")
        print(f"  Giá đóng cửa S_T = €{S_T:,.4f}")

    # ── Bước 8: Tính P&L (Lời / Lỗ) ──────────────────────────────────────────
    if S_T is not None:
        # Payoff của Call Option: max(S_T - K, 0)
        payoff_per_share = max(S_T - K, 0.0)
        total_payoff     = payoff_per_share * LOT_SIZE * N_CONTRACTS
        pnl              = total_payoff - total_cost
        pnl_pct          = (pnl / total_cost) * 100 if total_cost > 0 else 0.0
        result_label     = "LỜI ✅" if pnl >= 0 else "LỖ ❌"

        # Điểm hòa vốn (Break-even price)
        breakeven = K + call_premium
    else:
        payoff_per_share = total_payoff = pnl = pnl_pct = None
        result_label = "Chưa xác định"
        breakeven = K + call_premium

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 9 – VẼ VÀ HIỂN THỊ BIỂU ĐỒ NẾN TRƯỚC KHI IN BÁO CÁO
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n[BƯỚC 9] Đang vẽ biểu đồ nến... (sẽ mở trong trình duyệt)")

    # Lọc dữ liệu hiển thị: từ start_date đến expiration_date + 10 ngày
    fig_candle = build_candlestick_chart(
        df_chart=df_all,
        ticker=TICKER,
        start_date=actual_start,
        expiration_date=expiration_date,
        S0=S0,
        K=K,
        call_premium=call_premium,
    )

    # ← HIỂN THỊ BIỂU ĐỒ TRƯỚC (mở trình duyệt, người xem nhìn hình ảnh trước)
    fig_candle.show()

    # ══════════════════════════════════════════════════════════════════════════
    # BƯỚC 10 – IN "BÁO CÁO KIỂM THỬ" (SAU KHI ĐÃ HIỂN THỊ BIỂU ĐỒ)
    # ══════════════════════════════════════════════════════════════════════════
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + "  BÁO CÁO KIỂM THỬ MÔ PHỎNG QUYỀN CHỌN MUA (CALL OPTION)  ".center(68) + "║")
    print("╠" + "═" * 68 + "╣")

    print(f"║  {'Mã cổ phiếu':<30}: {TICKER:<35}║")
    print(f"║  {'Mô hình định giá':<30}: {'Black-Scholes-Merton (BSM)':<35}║")
    print(f"║  {'Loại quyền chọn':<30}: {'Call Option (Quyền chọn Mua)':<35}║")
    print(f"║  {'Kiểu quyền chọn':<30}: {'European (Kiểu Châu Âu)':<35}║")

    print("╠" + "═" * 68 + "╣")
    print("║" + "  📌 THAM SỐ ĐẦU VÀO  ".center(68) + "║")
    print("╠" + "═" * 68 + "╣")

    print(f"║  {'Ngày bắt đầu (Entry Date)':<30}: {str(actual_start):<35}║")
    print(f"║  {'Ngày đáo hạn (Expiry Date)':<30}: {str(expiration_date):<35}║")
    print(f"║  {'Kỳ hạn (Maturity)':<30}: {f'{MATURITY_DAYS} ngày lịch (~{T_years:.4f} năm)':<35}║")
    print(f"║  {'S₀ – Giá cổ phiếu ban đầu':<30}: {f'€{S0:,.4f}':<35}║")
    print(f"║  {'K  – Giá thực hiện (+1σ)':<30}: {f'€{K:,.4f}':<35}║")
    print(f"║  {'σ  – Biến động lịch sử (năm)':<30}: {f'{sigma*100:.2f}%  (window={HV_WINDOW}d)':<35}║")
    print(f"║  {'r  – Lãi suất phi rủi ro ECB':<30}: {f'{r*100:.2f}%':<35}║")

    print("╠" + "═" * 68 + "╣")
    print("║" + "  📊 KẾT QUẢ ĐỊNH GIÁ BSM  ".center(68) + "║")
    print("╠" + "═" * 68 + "╣")

    print(f"║  {'d₁':<30}: {bsm['d1']:>10.6f}{'':<24}║")
    print(f"║  {'d₂':<30}: {bsm['d2']:>10.6f}{'':<24}║")
    print(f"║  {'N(d₁) – Xác suất delta':<30}: {bsm['Nd1'] if 'Nd1' in bsm else norm.cdf(bsm['d1']):>10.6f}{'':<24}║")
    print(f"║  {'N(d₂) – Xác suất trung hòa rủi ro':<30}: {norm.cdf(bsm['d2']):>10.6f}{'':<24}║")
    print(f"║  {'Giá Call Option (BSM)':<30}: {f'€{call_premium:,.4f} / cổ phiếu':<35}║")
    print(f"║  {'Tổng phí quyền chọn':<30}: {f'€{total_cost:,.2f} ({N_CONTRACTS} HĐ × {LOT_SIZE} CP)':<35}║")
    print(f"║  {'Giá hòa vốn (Break-even)':<30}: {f'€{breakeven:,.4f}':<35}║")

    print("╠" + "═" * 68 + "╣")
    print("║" + "  🔢 GREEKS (ĐỘ NHẠY)  ".center(68) + "║")
    print("╠" + "═" * 68 + "╣")

    print(f"║  {'Δ Delta (∂C/∂S)':<30}: {bsm['delta']:>10.6f}  {'(giá trị Call thay đổi mỗi €1 S)':<20}║")
    print(f"║  {'Γ Gamma (∂²C/∂S²)':<30}: {bsm['gamma']:>10.6f}  {'(Delta thay đổi mỗi €1 S)':<20}║")
    print(f"║  {'Θ Theta (∂C/∂t / ngày)':<30}: {bsm['theta']:>10.6f}  {'(phí thời gian mỗi ngày)':<20}║")
    print(f"║  {'ν Vega  (∂C/∂σ / 1%)':<30}: {bsm['vega']:>10.6f}  {'(thay đổi mỗi 1% σ)':<20}║")
    print(f"║  {'ρ Rho   (∂C/∂r / 1%)':<30}: {bsm['rho']:>10.6f}  {'(thay đổi mỗi 1% r)':<20}║")

    print("╠" + "═" * 68 + "╣")
    print("║" + "  💰 KẾT QUẢ TẠI NGÀY ĐÁO HẠN  ".center(68) + "║")
    print("╠" + "═" * 68 + "╣")

    if S_T is not None:
        moneyness = "IN-THE-MONEY ✅" if S_T > K else "OUT-OF-THE-MONEY ❌"
        print(f"║  {'Giá thực tế S_T tại đáo hạn':<30}: {f'€{S_T:,.4f}':<35}║")
        print(f"║  {'Trạng thái quyền chọn':<30}: {moneyness:<35}║")
        print(f"║  {'Payoff = max(S_T − K, 0)':<30}: {f'€{payoff_per_share:,.4f} / cổ phiếu':<35}║")
        print(f"║  {'Tổng Payoff':<30}: {f'€{total_payoff:,.2f}':<35}║")
        print(f"║  {'Tổng phí đã bỏ ra (Premium)':<30}: {f'€{total_cost:,.2f}':<35}║")
        print("╠" + "═" * 68 + "╣")
        pnl_display = f"€{pnl:+,.2f}  ({pnl_pct:+.2f}%)"
        print(f"║  {'KẾT QUẢ CUỐI CÙNG':<30}: {f'{result_label} | {pnl_display}':<35}║")
    else:
        print(f"║  {'Giá tại đáo hạn':<30}: {'Chưa có dữ liệu':<35}║")
        print(f"║  {'Lưu ý':<30}: {'Kỳ hạn chưa đến hoặc dữ liệu thiếu':<35}║")

    print("╠" + "═" * 68 + "╣")
    print("║" + "  📝 NHẬN XÉT  ".center(68) + "║")
    print("╠" + "═" * 68 + "╣")

    if S_T is not None and pnl is not None:
        if pnl > 0:
            print(f"║  Quyền chọn MUA đáo hạn IN-THE-MONEY: S_T (€{S_T:,.2f}) > K (€{K:,.2f}){'':<4}║")
            print(f"║  Nhà đầu tư thực thi quyền, thu lợi ròng €{pnl:,.2f} ({pnl_pct:+.2f}%){'':<9}║")
        else:
            print(f"║  Quyền chọn MUA đáo hạn OUT-OF-THE-MONEY: S_T (€{S_T:,.2f}) ≤ K (€{K:,.2f}){'':<1}║")
            print(f"║  Nhà đầu tư bỏ quyền, tổn thất tối đa = phí premium €{total_cost:,.2f}{'':<6}║")
    else:
        print(f"║  {'Chưa thể tính P&L – vui lòng chạy lại sau ngày đáo hạn.':<66}║")

    print(f"║  {'─' * 66}║")
    print(f"║  {'⚠  Mô phỏng chỉ mang tính học thuật, không phải tư vấn đầu tư.':<66}║")
    print("╚" + "═" * 68 + "╝")


# ══════════════════════════════════════════════════════════════════════════════
# ĐIỂM VÀO CHƯƠNG TRÌNH
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()