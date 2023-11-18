import streamlit as st
import numpy as np
from scipy.stats import binom
from scipy.stats import norm
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.header('Binomial Distribution 二項分布', divider='rainbow')

st.latex(r'''
    P(Y=y) = {}_n\mathrm{C}_r p^y q^{n-y}, \hspace{5mm} y=0,1,...,n
    ''')

st.latex(
    r'''E[Y] = np, \hspace{5mm} V[Y] = npq, \hspace{5mm} G(s) = E[s^Y] = (ps+q)^n''')

col1, col2 = st.columns(2)

with col1:
    n = st.number_input("n", min_value=1, value=5)

with col2:
    p = st.number_input("p", min_value=0.01, max_value=0.99, value=0.4)

x_binom = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
mean_binom, var_binom = binom.stats(n, p, moments='mv')

st.write(f'E[Y]={mean_binom}')
st.write(f'V[Y]={var_binom}')

y = st.slider("y", min_value=0., max_value=float(n),
              value=mean_binom, step=0.1)

x_norm = np.linspace(norm.ppf(0.01, mean_binom, var_binom**0.5),
                     norm.ppf(0.99, mean_binom, var_binom**0.5), 100)

fig = make_subplots(rows=2, cols=1,
                    subplot_titles=("Probability Mass Function / Probability Density Function",
                                    "Cumulative Distribution Function"),
                    shared_xaxes=True,
                    vertical_spacing=0.08)

fig.add_trace(go.Scatter(x=x_binom, y=binom.pmf(x_binom, n, p),
                         mode='markers', name='binom',
                         marker=dict(size=12, line=dict(width=2))), row=1, col=1)

fig.add_trace(go.Scatter(x=x_norm, y=norm.pdf(x_norm, mean_binom, var_binom**0.5),
                         mode='lines', name='norm'), row=1, col=1)

fig.add_trace(go.Scatter(x=x_binom, y=binom.cdf(x_binom, n, p),
                         mode='markers', name='binom',
                         marker=dict(size=12, line=dict(width=2))), row=2, col=1)

fig.add_trace(go.Scatter(x=x_norm, y=norm.cdf(x_norm, mean_binom, var_binom**0.5),
                         mode='lines', name='norm'), row=2, col=1)

fig.add_trace(go.Scatter(x=x_norm, y=norm.cdf(x_norm+0.5, mean_binom, var_binom**0.5),
                         mode='lines', name='continuity correction'), row=2, col=1)

fig.add_trace(go.Scatter(x=[y, y],
                         y=[0, norm.pdf(y, mean_binom, var_binom**0.5)],
                         mode='lines', name='y',
                         line=dict(color='red', dash='dash')), row=1, col=1)

fig.add_trace(go.Scatter(x=[y, y],
                         y=[0, 1],
                         mode='lines', name='y',
                         line=dict(color='red', dash='dash')), row=2, col=1)

fig.update_layout(height=700)

st.plotly_chart(fig, theme="streamlit", use_container_width=True)
