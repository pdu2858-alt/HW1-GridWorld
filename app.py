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
        st.session_state.phase = 'set_start' # set_start -> set_end -> set_obstacles -> ready
    if 'policy' not in st.session_state:
        st.session_state.policy = None # 'random' 或 'optimal'
    if 'V' not in st.session_state:
        st.session_state.V = None
    if 'action_grid' not in st.session_state:
        st.session_state.action_grid = None
    if 'path' not in st.session_state:
        st.session_state.path = []

def reset_env():
    """重置地圖與演算法狀態"""
    st.session_state.start = None
    st.session_state.end = None
    st.session_state.obstacles = []
    st.session_state.phase = 'set_start'
    st.session_state.policy = None
    st.session_state.V = None
    st.session_state.action_grid = None
    st.session_state.path = []

def handle_click(r, c):
    """處理網格點擊邏輯"""
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
    """從起點開始根據政策追蹤最佳路徑"""
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
        if next_s != st.session_state.end:
            path.append(next_s)
        visited.add(next_s)
        curr = next_s
    return path

def policy_evaluation(n):
    """HW1-2: 策略評估"""
    V = np.zeros((n, n))
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
    """HW1-3: 價值迭代算法"""
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
                best_value = max(action_values)
                new_V[r, c] = best_value
                delta = max(delta, abs(best_value - V[r, c]))
        V = new_V
        if delta < theta:
            break
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

# --- UI ---
init_state()
active_color = "#00FF00" if st.session_state.policy == 'optimal' else "#00ffcc"
path_color = "#FFD700" # 金色代表路徑

st.markdown(f"""
    <style>
    [data-testid="column"] {{
        padding: 0px !important;
        margin: 0px !important;
    }}
    [data-testid="stHorizontalBlock"] {{
        gap: 0px !important;
    }}
    .stButton > button {{
        width: 100% !important;
        height: 60px !important;
        border-radius: 0px !important;
        padding: 0 !important;
        font-size: 14px !important;
        line-height: 1.2 !important;
        border: 0.1px solid #333 !important;
        background-color: #1e1e1e !important;
        color: {active_color} !important;
        margin: 0 !important;
    }}
    .stButton > button:hover {{
        background-color: #333 !important;
    }}
    .stButton > button:disabled {{
        background-color: #121212 !important;
        color: {active_color} !important;
        opacity: 1 !important;
    }}
    /* 路徑高亮 */
    .path-highlight > div > button {{
        background-color: #2e3b23 !important;
        border: 1px solid {path_color} !important;
        color: {path_color} !important;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("網格地圖 RL 導航器")

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
        if st.button("🎲 HW1-2: 隨機策略與評估"):
            policy_evaluation(st.session_state.n)
        if st.button("🧠 HW1-3: 價值迭代 (最佳策略)"):
            value_iteration(st.session_state.n)
    else:
        st.warning("請先完成地圖設定！")

phases_zhtw = {
    'set_start': "🟢 設定起點",
    'set_end': "🔴 設定終點",
    'set_obstacles': f"⬛ 設定障礙物 (還需 {st.session_state.n - 2 - len(st.session_state.obstacles)} 個)",
    'ready': "✅ 設定完成！"
}
st.subheader(phases_zhtw[st.session_state.phase])

for r in range(st.session_state.n):
    cols = st.columns(st.session_state.n)
    for c in range(st.session_state.n):
        pos = (r, c)
        cell_text = "⬜"
        is_disabled = False
        is_path = pos in st.session_state.path
        
        if pos == st.session_state.start:
            if st.session_state.policy:
                action_sym = ACTIONS[st.session_state.action_grid[r, c]][1]
                cell_text = f"🟢\n{action_sym}"
            else:
                cell_text = "🟢"
        elif pos == st.session_state.end:
            cell_text = "🔴"
        elif pos in st.session_state.obstacles:
            cell_text = "⬛"
        else:
            if st.session_state.policy:
                action_sym = ACTIONS[st.session_state.action_grid[r, c]][1]
                val = st.session_state.V[r, c]
                cell_text = f"{action_sym}\n{val:.1f}"
                is_disabled = True
        
        button_key = f"btn_{r}_{c}"
        if is_path:
            with cols[c]:
                st.markdown('<div class="path-highlight">', unsafe_allow_html=True)
                st.button(cell_text, key=button_key, disabled=is_disabled, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            if cols[c].button(cell_text, key=button_key, disabled=is_disabled, use_container_width=True):
                handle_click(r, c)
                st.rerun()

if st.session_state.policy:
    st.divider()
    status_label = "隨機策略" if st.session_state.policy == 'random' else "最佳策略"
    st.markdown(f"**當前：** <span style='color:{active_color}; font-weight:bold;'>{status_label}</span>", unsafe_allow_html=True)
    st.markdown(f"**$V(s)$ 狀態價值矩陣：**")
    st.dataframe(st.session_state.V)
