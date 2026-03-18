import streamlit as st
import numpy as np

# --- 參數與環境設定 ---
ACTIONS = [(np.array([-1, 0]), '↑'), (np.array([1, 0]), '↓'), 
           (np.array([0, -1]), '←'), (np.array([0, 1]), '→')]
GAMMA = 0.9  # 折扣因子
REWARD_GOAL = 100
REWARD_STEP = -1
REWARD_OBSTACLE = -10

def init_state():
    """初始化 Session State"""
    if 'n' not in st.session_state:
        st.session_state.n = 5
    if 'start' not in st.session_state:
        st.session_state.start = None
    if 'end' not in st.session_state:
        st.session_state.end = None
    if 'obstacles' not in st.session_state:
        st.session_state.obstacles = []
    if 'phase' not in st.session_state:
        st.session_state.phase = 'set_start'
    if 'policy' not in st.session_state:
        st.session_state.policy = None # 'random' 或 'optimal'
    if 'V' not in st.session_state:
        st.session_state.V = None
    if 'action_grid' not in st.session_state:
        st.session_state.action_grid = None
    if 'path' not in st.session_state:
        st.session_state.path = []

def reset_env():
    """重置環境"""
    st.session_state.start = None
    st.session_state.end = None
    st.session_state.obstacles = []
    st.session_state.phase = 'set_start'
    st.session_state.policy = None
    st.session_state.V = None
    st.session_state.action_grid = None
    st.session_state.path = []

def handle_click(r, c):
    """點擊網格邏輯"""
    pos = (r, c)
    if st.session_state.phase == 'set_start':
        st.session_state.start = pos
        st.session_state.phase = 'set_end'
    elif st.session_state.phase == 'set_end':
        if pos != st.session_state.start:
            st.session_state.end = pos
            st.session_state.phase = 'set_obstacles'
    elif st.session_state.phase == 'set_obstacles':
        if pos != st.session_state.start and pos != st.session_state.end:
            if pos not in st.session_state.obstacles:
                st.session_state.obstacles.append(pos)
                if len(st.session_state.obstacles) >= st.session_state.n - 2:
                    st.session_state.phase = 'ready'

def get_next_state(s, a_idx, n):
    """取得下一個狀態與回報"""
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
    """根據當前政策從起點追蹤路徑"""
    if st.session_state.action_grid is None or st.session_state.start is None:
        return []
    path = []
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
        if curr == st.session_state.end:
            break
    return path

def policy_evaluation(n):
    """HW1-2: 隨機策略評估"""
    V = np.zeros((n, n))
    # 隨機行動顯示
    action_grid = np.random.randint(0, 4, size=(n, n))
    theta = 1e-4
    while True:
        delta = 0
        new_V = np.copy(V)
        for r in range(n):
            for c in range(n):
                s = (r, c)
                if s == st.session_state.end or s in st.session_state.obstacles:
                    continue
                v_s = 0
                for a_idx in range(4):
                    next_s, reward = get_next_state(s, a_idx, n)
                    v_s += 0.25 * (reward + GAMMA * V[next_s[0], next_s[1]])
                new_V[r, c] = v_s
                delta = max(delta, abs(v_s - V[r, c]))
        V = new_V
        if delta < theta:
            break
    st.session_state.V = V
    st.session_state.action_grid = action_grid
    st.session_state.policy = 'random'
    st.session_state.path = calculate_path()

def value_iteration(n):
    """HW1-3: 價值迭代演算法"""
    V = np.zeros((n, n))
    action_grid = np.zeros((n, n), dtype=int)
    theta = 1e-4
    # 1. 價值迭代計算 V*
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
    # 2. 推導最佳政策 (Greedy Policy)
    for r in range(n):
        for c in range(n):
            s = (r, c)
            if s == st.session_state.end or s in st.session_state.obstacles:
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

# CSS 美化：移除間隙、定義狀態配色
st.markdown("""
    <style>
    [data-testid="column"] { padding: 0px !important; margin: 0px !important; }
    [data-testid="stHorizontalBlock"] { gap: 0px !important; }
    .stButton > button {
        width: 100% !important;
        height: 70px !important;
        border-radius: 0px !important;
        border: 0.1px solid #444 !important;
        background-color: #262626 !important;
        color: white !important;
        font-size: 14px !important;
        line-height: 1.2 !important;
        margin: 0 !important;
        display: block !important;
    }
    /* 起點 */
    .start-node > div > button { background-color: #2E7D32 !important; color: white !important; font-weight: bold !important; }
    /* 終點 */
    .end-node > div > button { background-color: #C62828 !important; color: white !important; font-weight: bold !important; }
    /* 障礙物 */
    .obstacle-node > div > button { background-color: #000000 !important; color: #555 !important; }
    /* 最佳路徑高亮 */
    .path-node > div > button { background-color: #FF8F00 !important; color: black !important; font-weight: bold !important; border: 1px solid gold !important; }
    
    .stButton > button p { margin: 0 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("網格地圖 RL 導航器 (HW1)")

with st.sidebar:
    st.header("環境設定")
    new_n = st.slider("選擇網格維度 (n)", 5, 10, st.session_state.n)
    if new_n != st.session_state.n:
        st.session_state.n = new_n
        reset_env()
        st.rerun()
    st.button("🔄 重置地圖", on_click=reset_env)
    
    st.header("演算法操作")
    if st.session_state.phase == 'ready':
        if st.button("🎲 HW1-2: 隨機策略評估"):
            policy_evaluation(st.session_state.n)
        if st.button("🧠 HW1-3: 價值迭代 (最佳政策)"):
            value_iteration(st.session_state.n)
    else:
        st.warning("請先完成地圖設定")

# 狀態提示
phases_msg = {
    'set_start': "🟢 請點擊設定「起點」",
    'set_end': "🔴 請點擊設定「終點」",
    'set_obstacles': f"⬛ 請設定「障礙物」 (剩餘 {st.session_state.n - 2 - len(st.session_state.obstacles)} 個)",
    'ready': "✅ 地圖就緒，請執行演算法"
}
st.subheader(phases_msg[st.session_state.phase])

# 繪製網格
for r in range(st.session_state.n):
    cols = st.columns(st.session_state.n)
    for c in range(st.session_state.n):
        pos = (r, c)
        cell_text = ""
        node_class = ""
        is_disabled = False
        
        # 決定顯示內容與樣式
        if pos == st.session_state.start:
            node_class = "start-node"
            cell_text = "START"
            if st.session_state.policy:
                action_sym = ACTIONS[st.session_state.action_grid[r, c]][1]
                cell_text = f"START\n{action_sym}"
        elif pos == st.session_state.end:
            node_class = "end-node"
            cell_text = "GOAL"
        elif pos in st.session_state.obstacles:
            node_class = "obstacle-node"
            cell_text = "⬛"
        else:
            if st.session_state.policy:
                # 顯示價值函數 V(s) 與 行動箭頭
                action_sym = ACTIONS[st.session_state.action_grid[r, c]][1]
                v_val = st.session_state.V[r, c]
                cell_text = f"{action_sym}\n{v_val:.1f}"
                is_disabled = True
                if pos in st.session_state.path:
                    node_class = "path-node"
            else:
                cell_text = ""
        
        # 渲染按鈕
        with cols[c]:
            if node_class:
                st.markdown(f'<div class="{node_class}">', unsafe_allow_html=True)
                st.button(cell_text, key=f"btn_{r}_{c}", disabled=is_disabled, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                if st.button(cell_text, key=f"btn_{r}_{c}", disabled=is_disabled, use_container_width=True):
                    handle_click(r, c)
                    st.rerun()

# 顯示數值矩陣
if st.session_state.policy:
    st.divider()
    type_str = "隨機政策 (Random)" if st.session_state.policy == 'random' else "最佳政策 (Optimal)"
    st.markdown(f"**當前顯示模式：** {type_str}")
    st.markdown(f"**$V(s)$ 狀態價值矩陣：**")
    st.dataframe(st.session_state.V)
