import streamlit as st
import numpy as np

# --- 參數與環境設定 ---
ACTIONS = [(np.array([-1, 0]), '↑'), (np.array([1, 0]), '↓'), 
           (np.array([0, -1]), '←'), (np.array([0, 1]), '→')]
GAMMA = 0.9
REWARD_GOAL = 10.0
REWARD_STEP = -0.1
REWARD_OBSTACLE = -1.0

def init_state():
    """根據提示詞初始化設定"""
    if 'n' not in st.session_state:
        st.session_state.n = 7
    if 'start' not in st.session_state:
        st.session_state.start = (0, 0)
    if 'end' not in st.session_state:
        st.session_state.end = (6, 6)
    if 'obstacles' not in st.session_state:
        # 預設一些牆壁，使用者仍可透過介面調整
        st.session_state.obstacles = [(1,1), (1,2), (2,1), (4,4), (4,5), (5,4)]
    if 'phase' not in st.session_state:
        st.session_state.phase = 'ready'
    if 'policy' not in st.session_state:
        st.session_state.policy = None
    if 'V' not in st.session_state:
        st.session_state.V = np.zeros((7, 7))
    if 'action_grid' not in st.session_state:
        st.session_state.action_grid = np.zeros((7, 7), dtype=int)
    if 'path' not in st.session_state:
        st.session_state.path = []

def reset_env():
    st.session_state.start = (0, 0)
    st.session_state.end = (6, 6)
    st.session_state.obstacles = [(1,1), (1,2), (2,1), (4,4), (4,5), (5,4)]
    st.session_state.phase = 'ready'
    st.session_state.policy = None
    st.session_state.V = np.zeros((st.session_state.n, st.session_state.n))
    st.session_state.path = []

def get_next_state(s, a_idx, n):
    move = ACTIONS[a_idx][0]
    next_s = (s[0] + move[0], s[1] + move[1])
    if (next_s[0] < 0 or next_s[0] >= n or 
        next_s[1] < 0 or next_s[1] >= n or 
        next_s in st.session_state.obstacles):
        return s, REWARD_OBSTACLE
    if next_s == st.session_state.end:
        return next_s, REWARD_GOAL
    return next_s, REWARD_STEP

def calculate_path():
    if st.session_state.action_grid is None or st.session_state.start is None:
        return []
    path = [st.session_state.start]
    curr = st.session_state.start
    visited = {curr}
    for _ in range(st.session_state.n * st.session_state.n):
        if curr == st.session_state.end:
            break
        a_idx = st.session_state.action_grid[curr[0], curr[1]]
        next_s, _ = get_next_state(curr, a_idx, st.session_state.n)
        if next_s == curr or next_s in visited:
            break
        path.append(next_s)
        visited.add(next_s)
        curr = next_s
    return path

def value_iteration(n):
    V = np.zeros((n, n))
    action_grid = np.zeros((n, n), dtype=int)
    theta = 1e-4
    while True:
        delta = 0
        new_V = np.copy(V)
        for r in range(n):
            for c in range(n):
                s = (r, c)
                if s == st.session_state.end or s in st.session_state.obstacles:
                    continue
                action_values = []
                for a_idx in range(4):
                    next_s, reward = get_next_state(s, a_idx, n)
                    action_values.append(reward + GAMMA * V[next_s[0], next_s[1]])
                new_V[r, c] = max(action_values)
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        V = new_V
        if delta < theta:
            break
    for r in range(n):
        for c in range(n):
            s = (r, c)
            if s == st.session_state.end:
                V[r, c] = REWARD_GOAL
                continue
            if s in st.session_state.obstacles:
                continue
            action_values = []
            for a_idx in range(4):
                next_s, reward = get_next_state(s, a_idx, n)
                action_values.append(reward + GAMMA * V[next_s[0], next_s[1]])
            action_grid[r, c] = np.argmax(action_values)
            
    st.session_state.V = V
    st.session_state.action_grid = action_grid
    st.session_state.policy = 'optimal'
    st.session_state.path = calculate_path()

# --- UI 呈現 ---
init_state()

st.markdown("""
    <style>
    /* 高解析度數據網格風格 */
    .grid-container {
        border: 3px solid black;
        display: inline-block;
        background-color: #f0f0f0;
    }
    [data-testid="column"] {
        padding: 0px !important;
        margin: 0px !important;
    }
    [data-testid="stHorizontalBlock"] {
        gap: 0px !important;
        border-bottom: 0.5px solid #ccc;
    }
    
    /* 格子基本樣式 */
    .stButton > button {
        width: 100% !important;
        height: 85px !important;
        border-radius: 0px !important;
        border: 0.5px solid #ccc !important;
        background-color: white !important;
        color: black !important;
        font-size: 13px !important;
        font-family: 'Arial', sans-serif !important;
        line-height: 1.1 !important;
        margin: 0 !important;
        transition: none !important;
    }
    
    /* 牆壁：黑色標記 WALL */
    .wall-node > div > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        font-weight: bold !important;
    }
    
    /* 最優路徑：鮮綠色高亮 (含 START, END) */
    .path-node > div > button {
        background-color: #00FF00 !important;
        color: black !important;
        font-weight: bold !important;
    }
    
    .stButton > button p { margin: 0 !important; }
    .stButton > button:hover { background-color: #f9f9f9 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("高解析度數據網格迷宮 - 價值迭代結果")

with st.sidebar:
    st.header("控制面板")
    if st.button("🧠 執行價值迭代 (Value Iteration)"):
        value_iteration(st.session_state.n)
    st.button("🔄 重置環境", on_click=reset_env)
    st.write("---")
    st.info("起點 (0,0), 終點 (6,6)\n綠色路徑代表最優導航。")

# 繪製網格
st.markdown('<div class="grid-container">', unsafe_allow_html=True)
for r in range(st.session_state.n):
    cols = st.columns(st.session_state.n)
    for c in range(st.session_state.n):
        pos = (r, c)
        cell_text = ""
        node_class = ""
        
        is_path = pos in st.session_state.path
        is_start = pos == st.session_state.start
        is_end = pos == st.session_state.end
        is_wall = pos in st.session_state.obstacles
        
        # 樣式決定
        if is_path or is_start or is_end:
            node_class = "path-node"
        elif is_wall:
            node_class = "wall-node"
        
        # 內容決定
        if is_wall:
            cell_text = "WALL\n-∞"
        elif is_end:
            # 終點包含最佳政策箭頭 (根據 V 計算)
            a_idx = st.session_state.action_grid[r, c] if st.session_state.policy else 1 # 預設下
            action_sym = ACTIONS[a_idx][1]
            v_val = st.session_state.V[r, c]
            cell_text = f"END\nV={v_val:.2f}\n{action_sym}"
        elif is_start:
            a_idx = st.session_state.action_grid[r, c] if st.session_state.policy else 3 # 預設右
            action_sym = ACTIONS[a_idx][1]
            v_val = st.session_state.V[r, c]
            cell_text = f"START\nV={v_val:.2f}\n{action_sym}"
        else:
            if st.session_state.policy:
                action_sym = ACTIONS[st.session_state.action_grid[r, c]][1]
                v_val = st.session_state.V[r, c]
                cell_text = f"V={v_val:.2f}\n{action_sym}"
            else:
                cell_text = ""

        with cols[c]:
            if node_class:
                st.markdown(f'<div class="{node_class}">', unsafe_allow_html=True)
                st.button(cell_text, key=f"btn_{r}_{c}", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.button(cell_text, key=f"btn_{r}_{c}", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.policy:
    st.success("最佳政策已推導。路徑由鮮綠色標記，箭頭指向最大化回報的方向。")
