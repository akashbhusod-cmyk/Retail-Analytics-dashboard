import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Retail Analytics & Customer Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

C = {"blue":"#185FA5","green":"#1D9E75","amber":"#BA7517","coral":"#D85A30","pink":"#D4537E","purple":"#534AB7","gray":"#888780"}
PALETTE = list(C.values())
LAYOUT  = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font=dict(family="system-ui,-apple-system,sans-serif",size=12),margin=dict(l=8,r=8,t=40,b=8),showlegend=True)
GRID    = dict(gridcolor="rgba(0,0,0,0.06)",zerolinecolor="rgba(0,0,0,0.06)")

st.markdown("""
<style>
header[data-testid="stHeader"],
div[data-testid="stToolbar"],
button[data-testid="collapsedControl"],
#MainMenu, footer { display: none !important; }

/* sidebar always open, no toggle */
section[data-testid="stSidebar"] {
    width: 260px !important;
    min-width: 260px !important;
    background: #f8f9fb !important;
}
section[data-testid="stSidebar"] > div {
    width: 260px !important;
    padding: 2rem 1.2rem 2rem 1.4rem !important;
}

.block-container { padding: 1.8rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── left panel ── */
.panel {
    background: #f8f9fb;
    border-right: 1px solid #e8e8e8;
    padding: 24px 16px 24px 16px;
    min-height: 100vh;
    position: fixed;
    top: 0; left: 0;
    width: 230px;
    overflow-y: auto;
    z-index: 100;
}
.panel-title {
    font-size: 11px; font-weight: 700; letter-spacing: 0.8px;
    text-transform: uppercase; color: #888; margin-bottom: 18px;
}
.panel-divider { border: none; border-top: 1px solid #e8e8e8; margin: 16px 0; }
.panel-meta-label { font-size: 11px; font-weight: 600; color: #555; margin: 10px 0 2px; }
.panel-meta-val   { font-size: 12px; color: #888; margin: 0 0 6px; }

/* ── main content area pushed right ── */
.main-area {
    margin-left: 242px;
    padding: 28px 28px 40px 28px;
}

/* ── header ── */
.dash-title {
    font-size: clamp(16px,2vw,22px); font-weight: 700;
    color: #111; margin: 0 0 4px; letter-spacing: -0.3px; line-height:1.3;
}
.dash-sub { font-size: 12px; color: #999; margin: 0 0 22px; }

/* ── KPI cards ── */
div[data-testid="stMetric"] {
    background: white; border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.07);
    border-left: 4px solid #185FA5;
    min-width: 0; overflow: hidden;
}
div[data-testid="stMetricLabel"] { font-size: 11px !important; color: #666 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
div[data-testid="stMetricValue"] { font-size: 16px !important; font-weight: 600 !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

h3 { font-size: 15px !important; font-weight: 600 !important; color: #222 !important; margin-top: 1.2rem !important; }

.insight {
    background: #F0F6FD; border-left: 3px solid #185FA5;
    border-radius: 4px; padding: 9px 13px;
    font-size: 13px; color: #1a3a5c; margin-bottom: 6px; line-height: 1.5;
}
.stPlotlyChart { width: 100% !important; }
div[data-testid="stDataFrame"] { overflow-x: auto !important; }
div[data-testid="stTabs"] > div:first-child { flex-wrap: wrap; gap: 4px; }
button[data-baseweb="tab"] { font-size: 13px !important; padding: 6px 12px !important; }
</style>
""", unsafe_allow_html=True)


# ── DATA ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data...")
def load_data():
    orders    = pd.read_csv("data/olist_orders_dataset.csv")
    items     = pd.read_csv("data/olist_order_items_dataset.csv")
    customers = pd.read_csv("data/olist_customers_dataset.csv")
    payments  = pd.read_csv("data/olist_order_payments_dataset.csv")
    products  = pd.read_csv("data/olist_products_dataset.csv")
    trans     = pd.read_csv("data/product_category_name_translation.csv")
    reviews   = pd.read_csv("data/olist_order_reviews_dataset.csv")
    sellers   = pd.read_csv("data/olist_sellers_dataset.csv")
    for col in ["order_purchase_timestamp","order_delivered_customer_date","order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    orders = orders[orders["order_status"]=="delivered"].copy()
    pay_agg = payments.groupby("order_id").agg(payment_value=("payment_value","sum"),payment_type=("payment_type","first"),payment_installments=("payment_installments","mean")).reset_index()
    products = products.merge(trans, on="product_category_name", how="left")
    rev_avg  = reviews.groupby("order_id")["review_score"].mean().reset_index()
    df = (orders
          .merge(items,on="order_id",how="left")
          .merge(customers,on="customer_id",how="left")
          .merge(pay_agg,on="order_id",how="left")
          .merge(products[["product_id","product_category_name_english"]],on="product_id",how="left")
          .merge(rev_avg,on="order_id",how="left")
          .merge(sellers[["seller_id","seller_state"]],on="seller_id",how="left"))
    df["revenue"]       = df["price"] + df["freight_value"]
    df["delivery_days"] = (df["order_delivered_customer_date"]-df["order_purchase_timestamp"]).dt.days
    df["on_time"]       = df["order_delivered_customer_date"] <= df["order_estimated_delivery_date"]
    df["year"]          = df["order_purchase_timestamp"].dt.year
    df["month"]         = df["order_purchase_timestamp"].dt.month
    df["year_month"]    = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    df["product_category_name_english"] = df["product_category_name_english"].fillna("other").str.replace("_"," ").str.title()
    return df

@st.cache_data(show_spinner="Computing RFM segments...")
def compute_rfm(df):
    snapshot = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer_unique_id").agg(recency=("order_purchase_timestamp",lambda x:(snapshot-x.max()).days),frequency=("order_id","nunique"),monetary=("revenue","sum")).reset_index()
    scaled = StandardScaler().fit_transform(rfm[["recency","frequency","monetary"]])
    rfm["cluster"] = KMeans(n_clusters=4,random_state=42,n_init=10).fit_predict(scaled)
    order = rfm.groupby("cluster")["monetary"].mean().sort_values(ascending=False).index
    rfm["segment"] = rfm["cluster"].map({c:l for c,l in zip(order,["Champions","Loyal","At Risk","Lost"])})
    return rfm

@st.cache_data(show_spinner="Computing cohort retention...")
def compute_cohort(df):
    c = df[["customer_unique_id","order_purchase_timestamp"]].drop_duplicates().copy()
    c["order_month"]  = c["order_purchase_timestamp"].dt.to_period("M")
    c["cohort_month"] = c.groupby("customer_unique_id")["order_month"].transform("min")
    c["period"]       = (c["order_month"]-c["cohort_month"]).apply(lambda x:x.n)
    pivot = c.groupby(["cohort_month","period"])["customer_unique_id"].nunique().unstack(fill_value=0)
    return (pivot.divide(pivot[0],axis=0)*100).iloc[:8,:7]


# ── CHARTS ────────────────────────────────────────────────────────────────

def _base(fig,title="",height=360):
    fig.update_layout(**LAYOUT,title=title,height=height); return fig

def chart_revenue_trend(df):
    m = df.groupby("year_month").agg(revenue=("revenue","sum"),orders=("order_id","nunique")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=m["year_month"],y=m["revenue"]/1000,name="Revenue (R$ K)",fill="tozeroy",line=dict(color=C["blue"],width=2.5),fillcolor="rgba(24,95,165,0.08)",mode="lines+markers",marker=dict(size=4)),secondary_y=False)
    fig.add_trace(go.Scatter(x=m["year_month"],y=m["orders"],name="Orders",line=dict(color=C["green"],width=2,dash="dot"),mode="lines+markers",marker=dict(size=4)),secondary_y=True)
    fig.update_layout(**LAYOUT,title="Monthly Revenue & Order Volume",height=380,legend=dict(orientation="h",y=1.1,x=0))
    fig.update_yaxes(title_text="Revenue (R$ K)",tickprefix="R$",ticksuffix="K",**GRID,secondary_y=False)
    fig.update_yaxes(title_text="Orders",showgrid=False,secondary_y=True)
    fig.update_xaxes(showgrid=False,tickangle=45,nticks=12)
    return fig

def chart_category_treemap(df):
    cat = df.groupby("product_category_name_english")["revenue"].sum().nlargest(15).reset_index()
    fig = px.treemap(cat,path=["product_category_name_english"],values="revenue",color="revenue",color_continuous_scale=["#C8DDF5",C["blue"]])
    fig.update_traces(textinfo="label+percent root"); fig.update_coloraxes(showscale=False)
    return _base(fig,"Revenue by Category (Top 15)",380)

def chart_order_status():
    counts = pd.Series({"Delivered":96478,"Shipped":1107,"Cancelled":625,"Unavailable":609,"Invoiced":314,"Processing":301})
    fig = px.pie(values=counts.values,names=counts.index,hole=0.55,color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="outside",textinfo="percent+label")
    return _base(fig,"Order Status Breakdown",360)

def chart_top_categories(df):
    top = df.groupby("product_category_name_english")["revenue"].sum().nlargest(12).sort_values().reset_index()
    fig = px.bar(top,x="revenue",y="product_category_name_english",orientation="h",color_discrete_sequence=[C["blue"]])
    fig.update_traces(marker_cornerradius=4); fig.update_xaxes(tickprefix="R$",**GRID); fig.update_yaxes(showgrid=False,tickfont=dict(size=11))
    return _base(fig,"Top 12 Categories by Revenue",400)

def chart_review_by_category(df):
    rev = (df.groupby("product_category_name_english").agg(avg_score=("review_score","mean"),count=("order_id","nunique")).reset_index().query("count >= 200").nlargest(12,"avg_score"))
    fig = px.bar(rev,x="avg_score",y="product_category_name_english",orientation="h",color="avg_score",color_continuous_scale=["#F7D9C4",C["green"]],text=rev["avg_score"].round(2))
    fig.update_coloraxes(showscale=False); fig.update_traces(textposition="outside",marker_cornerradius=4)
    fig.update_xaxes(range=[3,5.3],**GRID); fig.update_yaxes(showgrid=False,tickfont=dict(size=11))
    return _base(fig,"Avg Review Score by Category (min 200 orders)",400)

def chart_scatter_revenue_reviews(df):
    s = df.groupby("product_category_name_english").agg(revenue=("revenue","sum"),avg_score=("review_score","mean"),orders=("order_id","nunique")).reset_index().query("orders >= 100")
    fig = px.scatter(s,x="revenue",y="avg_score",size="orders",hover_name="product_category_name_english",color="revenue",color_continuous_scale=["#C8DDF5",C["blue"]],size_max=40)
    fig.update_coloraxes(showscale=False); fig.update_xaxes(tickprefix="R$",**GRID); fig.update_yaxes(**GRID,range=[2.5,5.3])
    return _base(fig,"Revenue vs Review Score  (bubble = order volume)",380)

def chart_rfm_donut(rfm):
    seg = rfm["segment"].value_counts().reset_index(); seg.columns=["segment","count"]
    fig = px.pie(seg,values="count",names="segment",hole=0.58,color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="outside",textinfo="percent+label")
    return _base(fig,"Customer Segments  (RFM Clustering)",360)

def chart_rfm_scatter(rfm):
    sample = rfm.sample(min(3000,len(rfm)),random_state=42)
    fig = px.scatter(sample,x="recency",y="monetary",color="segment",size="frequency",color_discrete_sequence=PALETTE,opacity=0.6,size_max=20)
    fig.update_xaxes(**GRID,title="Recency (days)"); fig.update_yaxes(**GRID,tickprefix="R$",title="Monetary (R$)")
    return _base(fig,"RFM Scatter  — Recency vs Monetary",380)

def chart_cohort(pct):
    fig = go.Figure(go.Heatmap(z=pct.values,x=[f"M+{i}" for i in pct.columns],y=[str(c) for c in pct.index],colorscale=[[0,"#E8F3FB"],[0.5,"#5BA3D9"],[1,"#042C53"]],text=np.round(pct.values,1),texttemplate="%{text}%",showscale=True,colorbar=dict(thickness=10,len=0.8)))
    fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False)
    return _base(fig,"Cohort Retention  (% of cohort still active)",360)

def chart_new_vs_returning(df):
    d = df[["customer_unique_id","order_id","year_month"]].drop_duplicates("order_id")
    first = d.groupby("customer_unique_id")["year_month"].min().reset_index(); first.columns=["customer_unique_id","first_month"]
    d = d.merge(first,on="customer_unique_id")
    d["type"] = d.apply(lambda r:"New" if r["year_month"]==r["first_month"] else "Returning",axis=1)
    m = d.groupby(["year_month","type"])["order_id"].nunique().reset_index()
    fig = px.bar(m,x="year_month",y="order_id",color="type",barmode="stack",color_discrete_map={"New":C["blue"],"Returning":C["green"]})
    fig.update_xaxes(showgrid=False,tickangle=45,title=""); fig.update_yaxes(**GRID,title="Customers")
    fig.update_layout(legend=dict(orientation="h",y=1.1,x=0))
    return _base(fig,"New vs Returning Customers — Monthly",360)

def chart_state_revenue(df):
    s = df.groupby("customer_state")["revenue"].sum().nlargest(12).sort_values().reset_index()
    fig = px.bar(s,x="revenue",y="customer_state",orientation="h",color_discrete_sequence=[C["blue"]])
    fig.update_traces(marker_cornerradius=4); fig.update_xaxes(tickprefix="R$",**GRID); fig.update_yaxes(showgrid=False)
    return _base(fig,"Top 12 States by Revenue",400)

def chart_delivery(df):
    d = df.groupby("customer_state").agg(avg_days=("delivery_days","median"),on_time_pct=("on_time","mean")).reset_index().dropna().nlargest(12,"avg_days")
    d["on_time_pct"]=(d["on_time_pct"]*100).round(1)
    fig = px.scatter(d,x="avg_days",y="on_time_pct",text="customer_state",size="avg_days",color="avg_days",color_continuous_scale=["#1D9E75","#BA7517",C["coral"]],size_max=25)
    fig.update_traces(textposition="top center",marker_line_width=0); fig.update_coloraxes(showscale=False)
    fig.update_xaxes(**GRID,title="Median Delivery Days"); fig.update_yaxes(**GRID,title="On-Time Rate (%)",ticksuffix="%")
    return _base(fig,"Delivery Days vs On-Time Rate by State",400)

def chart_payment_dist(df):
    p = df.groupby("payment_type")["payment_value"].sum().reset_index()
    p["payment_type"]=p["payment_type"].str.replace("_"," ").str.title()
    fig = px.pie(p,values="payment_value",names="payment_type",hole=0.55,color_discrete_sequence=PALETTE)
    fig.update_traces(textposition="outside",textinfo="percent+label")
    return _base(fig,"Payment Method Distribution",360)

def chart_installments(df):
    inst = df["payment_installments"].value_counts().reset_index().head(12)
    inst.columns=["installments","count"]; inst=inst.sort_values("installments")
    fig = px.bar(inst,x="installments",y="count",color_discrete_sequence=[C["purple"]])
    fig.update_traces(marker_cornerradius=4); fig.update_xaxes(showgrid=False,title="Number of Installments"); fig.update_yaxes(**GRID,title="Order Count")
    return _base(fig,"Payment Installments Distribution",340)

def two_col(left_fn,right_fn,h=400):
    c1,c2 = st.columns([1,1],gap="small")
    with c1:
        fig=left_fn(); fig.update_layout(height=h); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=right_fn(); fig.update_layout(height=h); st.plotly_chart(fig,use_container_width=True)


# ── LOAD ──────────────────────────────────────────────────────────────────

df_full = load_data()

# ── SIDEBAR FILTERS ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Filters")

    years      = sorted(df_full["year"].dropna().unique().tolist())
    sel_years  = st.multiselect("Year", years, default=years)

    sel_months = st.slider("Month Range", 1, 12, (1, 12))

    states_lst = ["All"] + sorted(df_full["customer_state"].dropna().unique().tolist())
    sel_state  = st.selectbox("Customer State", states_lst)

    cats_lst   = ["All"] + sorted(df_full["product_category_name_english"].dropna().unique().tolist())
    sel_cat    = st.selectbox("Product Category", cats_lst)

    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("Olist Brazilian E-Commerce")
    st.markdown("**Period**")
    st.caption("Sep 2016 – Oct 2018")
    st.markdown("**Records**")
    st.caption("99,441 orders")

# ── APPLY FILTERS ─────────────────────────────────────────────────────────

df = df_full.copy()
if sel_years:
    df = df[df["year"].isin(sel_years)]
df = df[df["month"].between(sel_months[0], sel_months[1])]
if sel_state != "All":
    df = df[df["customer_state"] == sel_state]
if sel_cat != "All":
    df = df[df["product_category_name_english"] == sel_cat]

# ── MAIN CONTENT ──────────────────────────────────────────────────────────

if True:

    st.markdown("""
<div class='dash-title'>Retail Analytics &amp; Customer Intelligence Dashboard</div>
<div class='dash-sub'>Olist Dataset &nbsp;&middot;&nbsp; Sep 2016 &ndash; Oct 2018 &nbsp;&middot;&nbsp; 96,478 Delivered Orders</div>
""", unsafe_allow_html=True)

    total_rev    = df["revenue"].sum()
    total_orders = df["order_id"].nunique()
    aov          = total_rev / max(total_orders, 1)
    avg_delivery = df["delivery_days"].median()
    avg_review   = df["review_score"].mean()
    on_time_pct  = df["on_time"].mean() * 100

    k1,k2,k3 = st.columns(3, gap="small")
    k1.metric("Total Revenue",   f"R$ {total_rev:,.0f}")
    k2.metric("Total Orders",    f"{total_orders:,}")
    k3.metric("Avg Order Value", f"R$ {aov:,.2f}")

    k4,k5,k6 = st.columns(3, gap="small")
    k4.metric("Median Delivery",  f"{avg_delivery:.1f} days")
    k5.metric("Avg Review Score", f"{avg_review:.2f} / 5.0")
    k6.metric("On-Time Delivery", f"{on_time_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1,tab2,tab3,tab4,tab5 = st.tabs(["Overview","Products","Customers","Geography","Payments"])

    with tab1:
        st.plotly_chart(chart_revenue_trend(df), use_container_width=True)
        two_col(lambda: chart_category_treemap(df), lambda: chart_order_status(), h=380)
        st.markdown("### Key Insights")
        best_month = df.groupby("year_month")["revenue"].sum().idxmax()
        best_cat   = df.groupby("product_category_name_english")["revenue"].sum().idxmax()
        best_state = df.groupby("customer_state")["revenue"].sum().idxmax()
        best_pct   = df[df["customer_state"]==best_state]["revenue"].sum()/total_rev*100
        pct_4star  = (df["review_score"]>=4).mean()*100
        for txt in [
            f"Peak revenue month: <strong>{best_month}</strong>",
            f"Top product category: <strong>{best_cat}</strong>",
            f"Top state: <strong>{best_state}</strong> — <strong>{best_pct:.1f}%</strong> of total revenue",
            f"<strong>{pct_4star:.1f}%</strong> of orders received a review score of 4 or 5",
        ]:
            st.markdown(f"<div class='insight'>{txt}</div>", unsafe_allow_html=True)

    with tab2:
        two_col(lambda: chart_top_categories(df), lambda: chart_review_by_category(df), h=400)
        st.plotly_chart(chart_scatter_revenue_reviews(df), use_container_width=True)
        st.markdown("### Product Category Table")
        tbl = (df.groupby("product_category_name_english").agg(revenue=("revenue","sum"),orders=("order_id","nunique"),avg_price=("price","mean"),avg_freight=("freight_value","mean"),avg_review=("review_score","mean")).reset_index().sort_values("revenue",ascending=False).head(20))
        tbl.columns=["Category","Revenue (R$)","Orders","Avg Price","Avg Freight","Avg Review"]
        for col in ["Revenue (R$)","Avg Price","Avg Freight"]: tbl[col]=tbl[col].map("R$ {:,.2f}".format)
        tbl["Avg Review"]=tbl["Avg Review"].map("{:.2f}".format)
        st.dataframe(tbl,use_container_width=True,hide_index=True)

    with tab3:
        rfm=compute_rfm(df); cohort_pct=compute_cohort(df)
        r1,r2,r3=st.columns(3,gap="small")
        r1.metric("Avg Recency",  f"{rfm['recency'].mean():.0f} days")
        r2.metric("Avg Frequency",f"{rfm['frequency'].mean():.2f} orders")
        r3.metric("Avg LTV",      f"R$ {rfm['monetary'].mean():,.2f}")
        st.markdown("<br>", unsafe_allow_html=True)
        two_col(lambda: chart_rfm_donut(rfm), lambda: chart_rfm_scatter(rfm), h=380)
        st.plotly_chart(chart_cohort(cohort_pct),use_container_width=True)
        st.plotly_chart(chart_new_vs_returning(df),use_container_width=True)
        st.markdown("### Segment Summary")
        seg_tbl=(rfm.groupby("segment").agg(customers=("customer_unique_id","count"),avg_recency=("recency","mean"),avg_frequency=("frequency","mean"),avg_ltv=("monetary","mean"),total_revenue=("monetary","sum")).reset_index().sort_values("total_revenue",ascending=False))
        seg_tbl.columns=["Segment","Customers","Avg Recency (days)","Avg Orders","Avg LTV (R$)","Total Revenue (R$)"]
        seg_tbl["Avg Recency (days)"]=seg_tbl["Avg Recency (days)"].map("{:.0f}".format)
        seg_tbl["Avg Orders"]=seg_tbl["Avg Orders"].map("{:.2f}".format)
        seg_tbl["Avg LTV (R$)"]=seg_tbl["Avg LTV (R$)"].map("R$ {:,.2f}".format)
        seg_tbl["Total Revenue (R$)"]=seg_tbl["Total Revenue (R$)"].map("R$ {:,.0f}".format)
        st.dataframe(seg_tbl,use_container_width=True,hide_index=True)

    with tab4:
        two_col(lambda: chart_state_revenue(df), lambda: chart_delivery(df), h=400)
        st.markdown("### State Performance Table")
        geo_tbl=(df.groupby("customer_state").agg(revenue=("revenue","sum"),orders=("order_id","nunique"),customers=("customer_unique_id","nunique"),avg_delivery=("delivery_days","median"),on_time=("on_time","mean"),avg_review=("review_score","mean")).reset_index().sort_values("revenue",ascending=False))
        geo_tbl.columns=["State","Revenue (R$)","Orders","Customers","Avg Delivery (days)","On-Time %","Avg Review"]
        geo_tbl["Revenue (R$)"]=geo_tbl["Revenue (R$)"].map("R$ {:,.0f}".format)
        geo_tbl["On-Time %"]=(geo_tbl["On-Time %"]*100).map("{:.1f}%".format)
        geo_tbl["Avg Delivery (days)"]=geo_tbl["Avg Delivery (days)"].map("{:.1f}".format)
        geo_tbl["Avg Review"]=geo_tbl["Avg Review"].map("{:.2f}".format)
        st.dataframe(geo_tbl,use_container_width=True,hide_index=True)

    with tab5:
        two_col(lambda: chart_payment_dist(df), lambda: chart_installments(df), h=360)
        pay_cat=(df.groupby(["product_category_name_english","payment_type"])["payment_value"].sum().reset_index().query("payment_type in ['credit_card','boleto','voucher','debit_card']"))
        pay_cat["payment_type"]=pay_cat["payment_type"].str.replace("_"," ").str.title()
        top_cats=pay_cat.groupby("product_category_name_english")["payment_value"].sum().nlargest(10).index
        pay_cat=pay_cat[pay_cat["product_category_name_english"].isin(top_cats)]
        fig_pc=px.bar(pay_cat,x="payment_value",y="product_category_name_english",color="payment_type",orientation="h",barmode="stack",color_discrete_sequence=PALETTE)
        fig_pc.update_xaxes(tickprefix="R$",**GRID); fig_pc.update_yaxes(showgrid=False,tickfont=dict(size=11))
        fig_pc.update_layout(**LAYOUT,title="Payment Method by Category (Top 10)",height=400,legend=dict(orientation="h",y=1.1,x=0))
        st.plotly_chart(fig_pc,use_container_width=True)

    st.markdown("---")
    st.caption("Built with Python · Streamlit · Plotly · Pandas · scikit-learn · Olist Dataset")