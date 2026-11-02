import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = load_data()
df["뇌졸중 여부"] = df["stroke"].map({1: "겪음", 0: "겪지 않음"})
GROUP_ORDER = ["겪지 않음", "겪음"]

st.title("🧠 탐색")

# ---- 1. 분포 히스토그램 ----
st.subheader("나이와 평균 혈당의 분포")
c1, c2 = st.columns(2)
with c1:
    fig = px.histogram(
        df, x="age", nbins=30, title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig)
with c2:
    fig = px.histogram(
        df, x="avg_glucose_level", nbins=30, title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig)

# ---- 2. 상자그림 ----
st.subheader("뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
c1, c2 = st.columns(2)
with c1:
    fig = px.box(
        df, x="뇌졸중 여부", y="age", color="뇌졸중 여부", title="나이",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        labels={"age": "나이"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
with c2:
    fig = px.box(
        df, x="뇌졸중 여부", y="avg_glucose_level", color="뇌졸중 여부",
        title="평균 혈당",
        category_orders={"뇌졸중 여부": GROUP_ORDER},
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

means = (
    df.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .round(2)
    .reindex(GROUP_ORDER)
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당의 평균"})
)
st.table(means)

# ---- 3. 고혈압·심장병별 뇌졸중 비율 ----
st.subheader("고혈압·심장병에 따른 뇌졸중 비율")


def rate_bar(column, title):
    rate = (
        df.groupby(column)["stroke"].mean().mul(100).round(2).reset_index()
    )
    rate["구분"] = rate[column].map({0: "없음", 1: "있음"})
    rate = rate.rename(columns={"stroke": "뇌졸중 비율(%)"})
    fig = px.bar(
        rate, x="구분", y="뇌졸중 비율(%)", text="뇌졸중 비율(%)", title=title,
        category_orders={"구분": ["없음", "있음"]},
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_title=title.split("에")[0])
    return fig


c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(rate_bar("hypertension", "고혈압 여부에 따른 뇌졸중 비율"))
with c2:
    st.plotly_chart(rate_bar("heart_disease", "심장병 여부에 따른 뇌졸중 비율"))

# ---- 4. bmi가 빈 사람들 ----
st.subheader("체질량지수(bmi)가 비어 있는 사람들")
missing = df[df["bmi"].isna()]
bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람들", "전체"],
        "사람 수": [len(missing), len(df)],
        "뇌졸중 사람 수": [int(missing["stroke"].sum()), int(df["stroke"].sum())],
        "뇌졸중 비율(%)": [
            round(missing["stroke"].mean() * 100, 2) if len(missing) else None,
            round(df["stroke"].mean() * 100, 2),
        ],
    }
)
st.dataframe(bmi_table, hide_index=True)

# ---- 5. 흡연 상태 ----
st.subheader("흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking, hide_index=True)
