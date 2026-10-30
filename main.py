import json
from pathlib import Path

import pandas as pd
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "기록.json"

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def load_record():
    """내_기록/기록.json 을 읽는다. 파일이 없거나 깨졌으면 빈 기록을 돌려준다."""
    try:
        data = json.loads(RECORD_FILE.read_text(encoding="utf-8"))
        meanings = data.get("우리말_뜻", {})
        source = data.get("데이터_출처", "")
        if not isinstance(meanings, dict) or not isinstance(source, str):
            raise ValueError("형식이 맞지 않아요")
        return {
            "우리말_뜻": {str(k): str(v) for k, v in meanings.items()},
            "데이터_출처": source,
        }
    except (OSError, ValueError, AttributeError):
        return {"우리말_뜻": {}, "데이터_출처": ""}


def save_record(record):
    try:
        RECORD_DIR.mkdir(exist_ok=True)
        RECORD_FILE.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except OSError:
        return False


def describe_values(s):
    values = s.dropna().unique()
    if not pd.api.types.is_numeric_dtype(s) or len(values) <= 10:
        return ", ".join(str(v) for v in sorted(values, key=str))
    return f"숫자 {s.min():g} ~ {s.max():g}"


df = load_data()

st.title("🧠 뇌졸중 예측 실습실")
st.write(
    "이 데이터는 사람마다 성별·나이·고혈압·심장병·혈당·체질량지수·흡연 상태 같은 "
    "정보와, 뇌졸중을 겪었는지(stroke)를 한 줄씩 적어 놓은 표예요. "
    "뇌졸중은 stroke가 1, 아니면 0이에요."
)

# ---- 큰 숫자 카드 네 개 ----
total = len(df)
n_stroke = int((df["stroke"] == 1).sum())
cards = [
    ("전체 사람 수", f"{total:,}명"),
    ("열 개수", f"{df.shape[1]}개"),
    ("stroke가 1인 사람 수", f"{n_stroke:,}명"),
    ("stroke가 1인 비율", f"{n_stroke / total:.1%}"),
]
for col, (label, value) in zip(st.columns(4), cards):
    with col:
        with st.container(border=True):
            st.metric(label, value)

# ---- 열 설명 표 ----
st.subheader("열 설명 표")
st.caption("‘우리말 뜻’ 칸은 교재를 보고 직접 채워 넣어요. 채우면 자동으로 저장돼요.")

if "saved_record" not in st.session_state:
    st.session_state.saved_record = load_record()
saved = st.session_state.saved_record

if "base_table" not in st.session_state:
    st.session_state.base_table = pd.DataFrame(
        {
            "열 이름": list(df.columns),
            "우리말 뜻": [saved["우리말_뜻"].get(c, "") for c in df.columns],
            "값의 종류": [describe_values(df[c]) for c in df.columns],
            "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
        }
    )

edited = st.data_editor(
    st.session_state.base_table,
    key="meaning_editor",
    hide_index=True,
    num_rows="fixed",
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", help="교재를 보고 적어요"),
    },
)

# ---- 처음 다섯 줄 ----
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5))

# ---- 데이터 출처 ----
st.subheader("데이터 출처")
if "source_text" not in st.session_state:
    st.session_state.source_text = saved["데이터_출처"]
source = st.text_area(
    "교재에 있는 출처 글을 적어 주세요 (입력 후 칸 밖을 누르거나 Ctrl+Enter)",
    key="source_text",
    height=120,
)

# ---- 저장 ----
new_record = {
    "우리말_뜻": {
        row["열 이름"]: (row["우리말 뜻"] or "") for _, row in edited.iterrows()
    },
    "데이터_출처": source or "",
}
if new_record != saved:
    if save_record(new_record):
        st.session_state.saved_record = new_record
        st.caption("✅ 내_기록 폴더에 저장했어요.")
    else:
        st.warning("내_기록 폴더에 저장하지 못했어요. 폴더 쓰기 권한을 확인해 주세요.")
else:
    st.caption("내_기록 폴더에 저장된 내용과 같아요.")
