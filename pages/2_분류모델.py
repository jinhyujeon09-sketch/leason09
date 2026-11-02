import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

KOR = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
ENG = {v: k for k, v in KOR.items()}
BINARY = {"hypertension", "heart_disease"}
DEFAULT_KOR = ["나이", "평균 혈당", "고혈압", "심장병"]

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

COLOR_NO, COLOR_YES = "#4c78a8", "#e45756"
REGION_NO, REGION_YES = "rgba(76,120,168,0.18)", "rgba(228,87,86,0.18)"
LEAF_NO, LEAF_YES = "#cfe2ff", "#f8d0cc"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧠 분류 모델")
st.write("뇌졸중(stroke=1)을 양성으로 두고, 두 가지 모델이 뇌졸중 여부를 얼마나 맞히는지 비교해요.")

# ---- 입력 속성 고르기 ----
chosen_kor = st.multiselect(
    "입력으로 사용할 속성", list(KOR.values()), default=DEFAULT_KOR
)
if len(chosen_kor) < 2:
    st.info("속성을 두 개 이상 골라 주세요.")
    st.stop()
cols = [c for c in KOR if KOR[c] in chosen_kor]

# ---- 나누기: 번호 순 정렬 → 열 명씩 묶어 앞 세 명은 테스트용 ----
df = load_data().sort_values("id", kind="stable").reset_index(drop=True)
is_test = (df.index.to_series() % 10) < 3
train, test = df[~is_test], df[is_test]
X_train, X_test = train[cols].copy(), test[cols].copy()
y_train, y_test = train["stroke"], test["stroke"]

st.caption(f"학습용 {len(train):,}명 · 테스트용 {len(test):,}명")

if "bmi" in cols:
    bmi_fill = X_train["bmi"].median()
    n_missing = int(df["bmi"].isna().sum())
    X_train["bmi"] = X_train["bmi"].fillna(bmi_fill)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_fill)
    st.caption(f"체질량지수의 빈 값 {n_missing}개는 훈련용의 중앙값 {bmi_fill:.1f}로 채웠어요.")

# ---- 모델 ----
scaler = StandardScaler().fit(X_train)  # 크기 맞추기는 학습용 사람으로만
lr = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
# 의사결정트리는 크기를 맞추지 않아도 같은 답이라, 질문을 원래 값으로 읽도록 그대로 학습해요.
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train, y_train
)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

scores = {
    LR_NAME: (
        lr.score(scaler.transform(X_train), y_train),
        lr.score(scaler.transform(X_test), y_test),
    ),
    TREE_NAME: (tree.score(X_train, y_train), tree.score(X_test, y_test)),
    BASE_NAME: (base.score(X_train, y_train), base.score(X_test, y_test)),
}

# ---- 정확도 카드 ----
st.subheader("정확도")
for col, (name, (train_acc, test_acc)) in zip(st.columns(3), scores.items()):
    with col:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.markdown(
                f"<div style='font-size:2.6rem;font-weight:700;line-height:1.2'>"
                f"{test_acc:.1%}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"훈련 데이터 {train_acc:.1%}  |  테스트 데이터 {test_acc:.1%}")

# ---- 산점도 + 경계 ----
st.subheader("테스트 데이터와 모델이 가른 모습")
cx, cy = st.columns(2)
x_kor = cx.selectbox("가로축", chosen_kor, index=0)
y_kor = cy.selectbox("세로축", chosen_kor, index=1)
if x_kor == y_kor:
    st.info("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
    st.stop()
xcol, ycol = ENG[x_kor], ENG[y_kor]


def axis_range(s, binary):
    lo, hi = float(s.min()), float(s.max())
    pad = 0.3 if binary else ((hi - lo) * 0.05 or 1.0)
    return lo - pad, hi + pad


N = 120
(xlo, xhi) = axis_range(X_test[xcol], xcol in BINARY)
(ylo, yhi) = axis_range(X_test[ycol], ycol in BINARY)
xs = [xlo + (xhi - xlo) * i / (N - 1) for i in range(N)]
ys = [ylo + (yhi - ylo) * i / (N - 1) for i in range(N)]

grid = pd.DataFrame({xcol: xs}).merge(pd.DataFrame({ycol: ys}), how="cross")
medians = X_test.median()
others = [c for c in cols if c not in (xcol, ycol)]
for c in others:
    grid[c] = medians[c]
grid = grid[cols]
grid = grid.assign(
    p=lr.predict_proba(scaler.transform(grid))[:, 1],
    t=tree.predict(grid),
)
P = grid.pivot(index=ycol, columns=xcol, values="p")
T = grid.pivot(index=ycol, columns=xcol, values="t")

fig = go.Figure()
fig.add_trace(
    go.Heatmap(
        x=list(T.columns), y=list(T.index), z=T.values, zmin=0, zmax=1,
        colorscale=[[0, REGION_NO], [1, REGION_YES]],
        showscale=False, hoverinfo="skip",
    )
)
p_min, p_max = float(P.values.min()), float(P.values.max())
line_visible = p_min < 0.5 < p_max
if line_visible:
    fig.add_trace(
        go.Contour(
            x=list(P.columns), y=list(P.index), z=P.values,
            contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
            line=dict(color="black", width=3),
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False, hoverinfo="skip", showlegend=False,
        )
    )

rng = random.Random(0)


def jitter(values, binary):
    return [v + rng.uniform(-0.08, 0.08) for v in values] if binary else list(values)


for label, val, color in [("아님(0)", 0, COLOR_NO), ("뇌졸중(1)", 1, COLOR_YES)]:
    mask = y_test == val
    fig.add_trace(
        go.Scatter(
            x=jitter(X_test.loc[mask, xcol], xcol in BINARY),
            y=jitter(X_test.loc[mask, ycol], ycol in BINARY),
            mode="markers", name=f"실제 {label}",
            marker=dict(color=color, size=6, opacity=0.7, line=dict(width=0.5, color="white")),
        )
    )

# 범례용 (배경 칸과 경계선)
present = set(T.values.ravel().tolist())
for val, name, color in [
    (0, f"{TREE_NAME.split('(')[0]}: 아님 칸", REGION_NO),
    (1, f"{TREE_NAME.split('(')[0]}: 뇌졸중 칸", REGION_YES),
]:
    if val in present:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", name=name,
                marker=dict(symbol="square", size=12, color=color),
            )
        )
if line_visible:
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"{LR_NAME.split('(')[0]} 0.5 경계선",
            line=dict(color="black", width=3),
        )
    )

fig.update_xaxes(range=[xlo, xhi], title=x_kor)
fig.update_yaxes(range=[ylo, yhi], title=y_kor)
fig.update_layout(legend_title_text="범례", height=520)
st.plotly_chart(fig)

if xcol in BINARY or ycol in BINARY:
    st.caption("고혈압·심장병처럼 0과 1뿐인 축은 점이 겹치지 않게 살짝 흩어서 찍었어요. 실제 값은 0 또는 1이에요.")
if others:
    fixed = ", ".join(f"{KOR[c]} {medians[c]:g}" for c in others)
    st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed}")
else:
    st.write("두 축 말고 고른 속성이 없어서, 따로 세워 둔 값은 없어요.")
if line_visible:
    st.write("검은 선은 로지스틱 회귀가 뇌졸중 확률 0.5로 가르는 선이에요.")
else:
    side = "아님" if p_max <= 0.5 else "뇌졸중"
    st.write(
        f"로지스틱 회귀의 0.5 선은 이 그림 밖에 있어요. 그림 안의 뇌졸중 확률은 "
        f"{p_min:.1%} ~ {p_max:.1%}라서, 그림 전체에서 '{side}'이라고 답해요."
    )

# ---- 의사결정트리 가지 그림 ----
st.subheader("의사결정트리가 던진 질문")
t = tree.tree_
path = tree.decision_path(X_train)
n_node = path.sum(axis=0).A1
n_pos = path.T.dot(y_train.values)

dot = [
    "digraph Tree {",
    'node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#555555"];',
    'edge [color="#555555"];',
]
leaf_answers = []
for i in range(t.node_count):
    n, p = int(n_node[i]), int(n_pos[i])
    rate = p / n if n else 0
    stats = f"사람 {n:,}명\\n뇌졸중 {p:,}명 ({rate:.1%})"
    left, right = t.children_left[i], t.children_right[i]
    if left == -1:
        answer = 1 if t.value[i][0][1] > t.value[i][0][0] else 0
        leaf_answers.append(answer)
        text = "뇌졸중" if answer else "아님"
        fill = LEAF_YES if answer else LEAF_NO
        dot.append(f'n{i} [label="답: {text}\\n{stats}", fillcolor="{fill}"];')
    else:
        feat = cols[t.feature[i]]
        if feat in BINARY:
            question = f"{KOR[feat]}이(가) 있나요?"
            left_label, right_label = "아니요", "예"
        else:
            question = f"{KOR[feat]} ≤ {t.threshold[i]:.1f}?"
            left_label, right_label = "예", "아니요"
        dot.append(f'n{i} [label="질문: {question}\\n{stats}"];')
        dot.append(f'n{i} -> n{left} [label="{left_label}"];')
        dot.append(f'n{i} -> n{right} [label="{right_label}"];')
dot.append("}")
st.graphviz_chart("\n".join(dot))

n_leaves = len(leaf_answers)
n_no = leaf_answers.count(0)
asked = {int(f) for f in t.feature if f >= 0}
asked_names = [KOR[c] for k, c in enumerate(cols) if k in asked]
st.write(f"답을 내는 마디는 모두 {n_leaves}칸이고, 그중 {n_no}칸이 '아님'이라고 답해요.")
if asked_names:
    st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것은 {', '.join(asked_names)}이에요.")
else:
    st.write("고른 속성 가운데 이 나무가 실제로 물은 것은 없어요. 질문 없이 바로 답했어요.")
