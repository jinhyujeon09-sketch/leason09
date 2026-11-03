import pandas as pd
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
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
# '분류 모델' 페이지에서 처음 골라져 있는 넷과 같은 입력 속성
cols = list(KOR)

LR_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"
NA_TEXT = "계산할 수 없음"
LIST_LIMIT = 20


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧠 채점표")
st.write("뇌졸중(stroke=1)을 양성으로 두고, 테스트 데이터로 모델의 답을 채점해요.")

# ---- '분류 모델' 페이지와 같은 방법: 읽기, 나누기, 학습 ----
df = load_data().sort_values("id", kind="stable").reset_index(drop=True)
is_test = (df.index.to_series() % 10) < 3  # 열 명씩 묶어 앞 세 명은 테스트용
train, test = df[~is_test], df[is_test]
X_train, X_test = train[cols].copy(), test[cols].copy()
y_train, y_test = train["stroke"], test["stroke"]

scaler = StandardScaler().fit(X_train)
lr = LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(
    X_train, y_train
)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

preds = {
    LR_NAME: pd.Series(lr.predict(scaler.transform(X_test)), index=test.index),
    TREE_NAME: pd.Series(tree.predict(X_test), index=test.index),
    BASE_NAME: pd.Series(base.predict(X_test), index=test.index),
}

st.caption("입력 속성: " + ", ".join(KOR.values()) + " (‘분류 모델’ 페이지에서 처음 골라져 있는 속성과 같아요)")
n_test = len(test)
n_pos = int((y_test == 1).sum())
st.write(f"테스트용 {n_test:,}명 중 뇌졸중인 사람은 {n_pos:,}명이에요.")


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
    total = tp + fn + fp + tn
    accuracy = (tp + tn) / total
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


all_counts = {name: counts(p) for name, p in preds.items()}

# ---- 혼동행렬 ----
st.subheader("혼동행렬 (행은 실제, 열은 예측)")


def confusion_table(c):
    row_yes = c["TP"] + c["FN"]
    row_no = c["FP"] + c["TN"]
    col_yes = c["TP"] + c["FP"]
    col_no = c["FN"] + c["TN"]
    return pd.DataFrame(
        {
            "예측: 뇌졸중": [
                f"TP · {c['TP']:,}명",
                f"FP · {c['FP']:,}명",
                f"{col_yes:,}명",
            ],
            "예측: 아님": [
                f"FN · {c['FN']:,}명",
                f"TN · {c['TN']:,}명",
                f"{col_no:,}명",
            ],
            "합계": [f"{row_yes:,}명", f"{row_no:,}명", f"{row_yes + row_no:,}명"],
        },
        index=["실제: 뇌졸중", "실제: 아님", "합계"],
    )


for col, name in zip(st.columns(2), [LR_NAME, TREE_NAME]):
    with col:
        st.markdown(f"**{name}**")
        st.table(confusion_table(all_counts[name]))
st.caption("TP: 뇌졸중을 뇌졸중이라고 맞힘 · FN: 뇌졸중을 놓침 · FP: 아닌데 뇌졸중이라고 함 · TN: 아님을 아님이라고 맞힘")

# ---- 지표 비교표 ----
st.subheader("네 가지 지표 비교")
rows = []
for name in [LR_NAME, TREE_NAME, BASE_NAME]:
    accuracy, recall, precision, f1 = metrics(all_counts[name])
    rows.append(
        {
            "모델": name,
            "정확도": fmt(accuracy),
            "재현율": fmt(recall),
            "정밀도": fmt(precision),
            "F1": fmt(f1),
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption(
    "정확도 = (TP+TN) ÷ 전체 · 재현율 = TP ÷ 실제 뇌졸중인 사람 수(TP+FN) · "
    "정밀도 = TP ÷ 뇌졸중이라 예측한 사람 수(TP+FP) · F1 = 2×정밀도×재현율 ÷ (정밀도+재현율)"
)

# ---- 놓친 환자 / 헛짚은 사람 ----
st.subheader("틀린 사람 살펴보기")
model_name = st.selectbox("어느 모델의 목록을 볼까요?", [LR_NAME, TREE_NAME])
pred = preds[model_name]


def person_list(mask):
    people = test[mask].sort_values("age", ascending=False, kind="stable")
    total = len(people)
    shown = people.head(LIST_LIMIT)
    table = pd.DataFrame(
        {
            "번호": shown["id"],
            "나이": shown["age"],
            "평균 혈당": shown["avg_glucose_level"],
            "고혈압": shown["hypertension"].map({0: "없음", 1: "있음"}),
            "심장병": shown["heart_disease"].map({0: "없음", 1: "있음"}),
        }
    )
    return total, table


missed_total, missed_table = person_list((y_test == 1) & (pred == 0))
wrong_total, wrong_table = person_list((y_test == 0) & (pred == 1))

left, right = st.columns(2)
with left:
    st.markdown(f"**놓친 환자** (실제 뇌졸중인데 아니라고 예측) — 전체 {missed_total:,}명")
    if missed_total > LIST_LIMIT:
        st.caption(f"나이가 많은 순서로 앞 {LIST_LIMIT}줄만 보여 줘요.")
    st.dataframe(missed_table, hide_index=True)
with right:
    st.markdown(f"**헛짚은 사람** (아닌데 뇌졸중이라고 예측) — 전체 {wrong_total:,}명")
    if wrong_total > LIST_LIMIT:
        st.caption(f"나이가 많은 순서로 앞 {LIST_LIMIT}줄만 보여 줘요.")
    st.dataframe(wrong_table, hide_index=True)

c = all_counts[model_name]
if missed_total == c["FN"] and wrong_total == c["FP"]:
    st.write(
        f"두 목록의 전체 사람 수({missed_total:,}명, {wrong_total:,}명)는 "
        f"혼동행렬의 FN {c['FN']:,}명, FP {c['FP']:,}명과 같아요."
    )
else:
    st.warning("목록의 사람 수와 혼동행렬의 FN·FP가 달라요. 코드를 다시 확인해 주세요.")
