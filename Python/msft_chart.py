import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

ticker = yf.Ticker("MSFT")
df = ticker.history(period="5y", interval="1d")
df.index = df.index.tz_localize(None)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    mode="lines",
    name="MSFT",
    line=dict(color="#00a8e0", width=1.5),
    hovertemplate="<b>%{x|%b %d, %Y}</b><br>Close: $%{y:.2f}<extra></extra>",
))

now = datetime.now()

fig.update_layout(
    title=dict(text="MSFT — Microsoft Corporation", font=dict(size=20)),
    xaxis=dict(
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1D",  step="day",   stepmode="backward"),
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=1,  label="YTD", step="year",  stepmode="todate"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(step="all", label="5Y"),
            ],
            bgcolor="#1e1e2e",
            activecolor="#00a8e0",
            font=dict(color="white"),
        ),
        rangeslider=dict(visible=True),
        type="date",
        range=[str(now - timedelta(days=365)), str(now)],
    ),
    yaxis=dict(
        title="Price (USD)",
        tickprefix="$",
        side="right",
    ),
    template="plotly_dark",
    hovermode="x unified",
    height=650,
    margin=dict(l=40, r=60, t=60, b=40),
)

fig.show()
