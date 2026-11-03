import random

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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
NA_TEXT = "계산할 수 없음"
BEFORE, AFTER = "가중치 맞추기 전", "가중치 맞춘 뒤"
COLOR_NO, COLOR_YES = "#4c78a8", "#e45756"
COLOR_BEFORE, COLOR_AFTER = "black", "#2ca02c"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧠 성능 높이기")
st.write(
    "뇌졸중(stroke=1)을 양성으로 두어요. 뇌졸중인 사람은 아주 드물어서, 모델이 '아님'만 "
    "답해도 정확도가 높게 나와요. 두 쪽이 학습에 미치는 힘(가중치)을 맞추면 결과가 어떻게 "
    "달라지는지 비교해 봐요."
)

# ---- 입력 속성 고르기 ----
chosen_kor = st.multiselect(
    "입력으로 사용할 속성", list(KOR.values()), default=DEFAULT_KOR
)
if len(chosen_kor) < 2:
    st.info("속성을 두 개 이상 골라 주세요.")
    st.stop()
cols = [c for c in KOR if KOR[c] in chosen_kor]

# ---- 읽기와 나누기 (채점표와 같은 방법) ----
df = load_data().sort_values("id", kind="stable").reset_index(drop=True)
is_test = (df.index.to_series() % 10) < 3  # 열 명씩 묶어 앞 세 명은 테스트용
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

# ---- 모델: 가중치 맞추기 전·후 ----
scaler = StandardScaler().fit(X_train)
Xs_train, Xs_test = scaler.transform(X_train), scaler.transform(X_test)

lr_before = LogisticRegression(max_iter=1000).fit(Xs_train, y_train)
lr_after = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xs_train, y_train)
tree_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train, y_train
)
tree_after = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42, class_weight="balanced"
).fit(X_train, y_train)

preds = {
    LR_NAME: (
        pd.Series(lr_before.predict(Xs_test), index=test.index),
        pd.Series(lr_after.predict(Xs_test), index=test.index),
    ),
    TREE_NAME: (
        pd.Series(tree_before.predict(X_test), index=test.index),
        pd.Series(tree_after.predict(X_test), index=test.index),
    ),
}


def counts(pred):
    actual_yes = y_test == 1
    pred_yes = pred == 1
    return {
        "TP": int((actual_yes & pred_yes).sum()),
        "FN": int((actual_yes & ~pred_yes).sum()),
        "FP": int((~actual_yes & pred_yes).sum()),
        "TN": int((~actual_yes & ~pred_yes).sum()),
    }


def metrics(c):
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    accuracy = (tp + tn) / (tp + fn + fp + tn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0  # 둘 다 0이면 F1도 0으로 둬요
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return accuracy, recall, precision, f1


def fmt(x):
    return NA_TEXT if x is None else f"{x:.1%}"


def compare_table(c_before, c_after):
    rows = {
        "TP (찾아낸 환자)": [f"{c_before['TP']:,}명", f"{c_after['TP']:,}명"],
        "FN (놓친 환자)": [f"{c_before['FN']:,}명", f"{c_after['FN']:,}명"],
        "FP (헛짚은 사람)": [f"{c_before['FP']:,}명", f"{c_after['FP']:,}명"],
        "TN (아님을 맞힘)": [f"{c_before['TN']:,}명", f"{c_after['TN']:,}명"],
    }
    mb, ma = metrics(c_before), metrics(c_after)
    for k, label in enumerate(["정확도", "재현율", "정밀도", "F1"]):
        rows[label] = [fmt(mb[k]), fmt(ma[k])]
    return pd.DataFrame(rows, index=[BEFORE, AFTER]).T


# ---- 전·후 비교표 ----
st.subheader("가중치를 맞추기 전과 맞춘 뒤")
st.caption(
    "가중치 맞추기: 드문 쪽(뇌졸중)의 한 사람을 더 무겁게 보아, 두 쪽이 학습에 미치는 힘을 같게 만들어요. "
    "테스트용 " + f"{len(test):,}명 중 뇌졸중인 사람은 {int((y_test == 1).sum()):,}명이에요."
)
for col, name in zip(st.columns(2), [LR_NAME, TREE_NAME]):
    before_pred, after_pred = preds[name]
    cb, ca = counts(before_pred), counts(after_pred)
    with col:
        st.markdown(f"**{name}**")
        st.table(compare_table(cb, ca))
        st.write(
            f"찾아낸 환자 {cb['TP']:,}명 → {ca['TP']:,}명, "
            f"놓친 환자 {cb['FN']:,}명 → {ca['FN']:,}명, "
            f"헛짚은 사람 {cb['FP']:,}명 → {ca['FP']:,}명"
        )

# ---- 그림 준비 ----
medians = X_test.median()
orig_coefs = {}
for key, model in [("before", lr_before), ("after", lr_after)]:
    w = model.coef_[0]
    c = [float(w[j] / scaler.scale_[j]) for j in range(len(cols))]
    b = float(model.intercept_[0]) - sum(
        float(w[j] * scaler.mean_[j] / scaler.scale_[j]) for j in range(len(cols))
    )
    orig_coefs[key] = (c, b)  # 원래 값 기준: b + Σ c·x = 0 이 0.5 경계


def axis_range(s, binary):
    lo, hi = float(s.min()), float(s.max())
    pad = 0.3 if binary else ((hi - lo) * 0.05 or 1.0)
    return lo - pad, hi + pad


def linspace(lo, hi, n):
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


rng = random.Random(0)


def jitter(values, binary):
    return [v + rng.uniform(-0.08, 0.08) for v in values] if binary else list(values)


# ---- 2차원 산점도 ----
st.subheader("두 속성으로 본 0.5 경계선")
cx, cy = st.columns(2)
x_kor = cx.selectbox("가로축", chosen_kor, index=0, key="x2d")
y_kor = cy.selectbox("세로축", chosen_kor, index=1, key="y2d")
if x_kor == y_kor:
    st.info("가로축과 세로축은 서로 다른 속성으로 골라 주세요.")
else:
    xcol, ycol = ENG[x_kor], ENG[y_kor]
    xlo, xhi = axis_range(X_test[xcol], xcol in BINARY)
    ylo, yhi = axis_range(X_test[ycol], ycol in BINARY)
    N = 120
    xs, ys = linspace(xlo, xhi, N), linspace(ylo, yhi, N)

    grid = pd.DataFrame({xcol: xs}).merge(pd.DataFrame({ycol: ys}), how="cross")
    others = [c for c in cols if c not in (xcol, ycol)]
    for c in others:
        grid[c] = medians[c]
    grid = grid[cols]
    grid_s = scaler.transform(grid)
    grid = grid.assign(
        p_before=lr_before.predict_proba(grid_s)[:, 1],
        p_after=lr_after.predict_proba(grid_s)[:, 1],
    )

    fig = go.Figure()
    line_notes = []
    legend_lines = []
    for key, label, color in [("before", BEFORE, COLOR_BEFORE), ("after", AFTER, COLOR_AFTER)]:
        P = grid.pivot(index=ycol, columns=xcol, values=f"p_{key}")
        p_min, p_max = float(P.values.min()), float(P.values.max())
        if p_min < 0.5 < p_max:
            fig.add_trace(
                go.Contour(
                    x=list(P.columns), y=list(P.index), z=P.values,
                    contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                    line=dict(color=color, width=3),
                    colorscale=[[0, color], [1, color]],
                    showscale=False, hoverinfo="skip", showlegend=False,
                )
            )
            legend_lines.append((label, color))
        else:
            side = "아님" if p_max <= 0.5 else "뇌졸중"
            line_notes.append(
                f"{label}의 0.5 선은 이 그림 밖에 있어요. 그림 안의 뇌졸중 확률은 "
                f"{p_min:.1%} ~ {p_max:.1%}라서, 그림 전체에서 '{side}'이라고 답해요."
            )

    for label, val, color in [("아님(0)", 0, COLOR_NO), ("뇌졸중(1)", 1, COLOR_YES)]:
        mask = y_test == val
        fig.add_trace(
            go.Scatter(
                x=jitter(X_test.loc[mask, xcol], xcol in BINARY),
                y=jitter(X_test.loc[mask, ycol], ycol in BINARY),
                mode="markers", name=f"실제 {label}",
                marker=dict(color=color, size=6, opacity=0.6, line=dict(width=0.5, color="white")),
            )
        )
    for label, color in legend_lines:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                name=f"{label} 0.5 경계선", line=dict(color=color, width=3),
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
    st.write(
        "선은 모두 로지스틱 회귀가 뇌졸중 확률 0.5로 가르는 선이에요. "
        f"검은 선은 {BEFORE}, 초록 선은 {AFTER}예요."
    )
    for note in line_notes:
        st.write(note)

# ---- 3차원 산점도 ----
if len(chosen_kor) >= 3:
    st.subheader("세 속성으로 본 0.5 평면")
    c1, c2, c3 = st.columns(3)
    a_kor = [
        c1.selectbox("가로축(x)", chosen_kor, index=0, key="x3d"),
        c2.selectbox("세로축(y)", chosen_kor, index=1, key="y3d"),
        c3.selectbox("높이축(z)", chosen_kor, index=2, key="z3d"),
    ]
    if len(set(a_kor)) < 3:
        st.info("세 축은 서로 다른 속성으로 골라 주세요.")
    else:
        a_cols = [ENG[k] for k in a_kor]
        ranges = {c: axis_range(X_test[c], c in BINARY) for c in a_cols}
        idx = {c: k for k, c in enumerate(cols)}
        others3 = [c for c in cols if c not in a_cols]

        def plane(key):
            """원래 값 기준 b + Σ c·x = 0 평면을 격자로 만든다. 그림 밖이면 None."""
            c, b = orig_coefs[key]
            fixed_part = b + sum(c[idx[o]] * float(medians[o]) for o in others3)
            s = max(a_cols, key=lambda col: abs(c[idx[col]]))
            if abs(c[idx[s]]) < 1e-12:
                return None
            u, v = [col for col in a_cols if col != s]
            lo, hi = ranges[s]
            M = 25
            us, vs = linspace(*ranges[u], M), linspace(*ranges[v], M)
            U, V, S = [], [], []
            visible = False
            for vv in vs:
                ru, rv, rs = [], [], []
                for uu in us:
                    val = -(fixed_part + c[idx[u]] * uu + c[idx[v]] * vv) / c[idx[s]]
                    if lo <= val <= hi:
                        ru.append(uu); rv.append(vv); rs.append(val)
                        visible = True
                    else:
                        ru.append(None); rv.append(None); rs.append(None)
                U.append(ru); V.append(rv); S.append(rs)
            if not visible:
                return None
            coords = {u: U, v: V, s: S}
            return coords[a_cols[0]], coords[a_cols[1]], coords[a_cols[2]]

        fig3 = go.Figure()
        notes3 = []
        for key, label, color in [("before", BEFORE, COLOR_BEFORE), ("after", AFTER, COLOR_AFTER)]:
            surf = plane(key)
            if surf is None:
                notes3.append(f"{label}의 0.5 평면은 이 그림 상자 밖에 있어요.")
                continue
            X3, Y3, Z3 = surf
            fig3.add_trace(
                go.Surface(
                    x=X3, y=Y3, z=Z3, opacity=0.4,
                    colorscale=[[0, color], [1, color]],
                    showscale=False, hoverinfo="skip", showlegend=False,
                )
            )
            fig3.add_trace(
                go.Scatter3d(
                    x=[None], y=[None], z=[None], mode="lines",
                    name=f"{label} 0.5 평면", line=dict(color=color, width=6),
                )
            )
        for label, val, color in [("아님(0)", 0, COLOR_NO), ("뇌졸중(1)", 1, COLOR_YES)]:
            mask = y_test == val
            fig3.add_trace(
                go.Scatter3d(
                    x=jitter(X_test.loc[mask, a_cols[0]], a_cols[0] in BINARY),
                    y=jitter(X_test.loc[mask, a_cols[1]], a_cols[1] in BINARY),
                    z=jitter(X_test.loc[mask, a_cols[2]], a_cols[2] in BINARY),
                    mode="markers", name=f"실제 {label}",
                    marker=dict(color=color, size=3, opacity=0.6),
                )
            )
        fig3.update_layout(
            height=650,
            legend_title_text="범례",
            scene=dict(
                xaxis=dict(title=a_kor[0], range=list(ranges[a_cols[0]])),
                yaxis=dict(title=a_kor[1], range=list(ranges[a_cols[1]])),
                zaxis=dict(title=a_kor[2], range=list(ranges[a_cols[2]])),
            ),
        )
        st.plotly_chart(fig3)
        st.caption("마우스로 끌어서 돌려 볼 수 있어요. 휠로 확대·축소할 수 있어요.")
        if any(c in BINARY for c in a_cols):
            st.caption("고혈압·심장병처럼 0과 1뿐인 축은 점이 겹치지 않게 살짝 흩어서 찍었어요.")
        if others3:
            fixed3 = ", ".join(f"{KOR[c]} {medians[c]:g}" for c in others3)
            st.write(f"세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했어요: {fixed3}")
        else:
            st.write("세 축 말고 고른 속성이 없어서, 따로 세워 둔 값은 없어요.")
        st.write(
            "평면은 모두 로지스틱 회귀가 뇌졸중 확률 0.5로 가르는 면이에요. "
            f"검은 평면은 {BEFORE}, 초록 평면은 {AFTER}예요."
        )
        for note in notes3:
            st.write(note)
else:
    st.caption("속성을 셋 이상 고르면 3차원 산점도와 0.5 평면도 볼 수 있어요.")
