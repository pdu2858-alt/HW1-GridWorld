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

def reset_env():
    """重置地圖與演算法狀態"""
    st.session_state.start = None
    st.session_state.end = None
    st.session_state.obstacles = []
    st.session_state.phase = 'set_start'
    st.session_state.policy = None
    st.session_state.V = None
    st.session_state.action_grid = None

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
                # 檢查是否達到 n-2 個障礙物
                if len(st.session_state.obstacles) >= st.session_state.n - 2:
                    st.session_state.phase = 'ready'

# --- 強化學習核心演算法 ---

def get_next_state(s, a_idx, n):
    """取得下一個狀態與回報"""
    move = ACTIONS[a_idx][0]
    next_s = (s[0] + move[0], s[1] + move[1])
    
    # 撞牆或撞障礙物，留在原地並給予負回報
    if (next_s[0] < 0 or next_s[0] >= n or 
        next_s[1] < 0 or next_s[1] >= n or 
        next_s in st.session_state.obstacles):
        return s, REWARD_OBSTACLE
    
    # 到達終點
    if next_s == st.session_state.end:
        return next_s, REWARD_GOAL
        
    return next_s, REWARD_STEP

def policy_evaluation(n):
    """HW1-2: 策略評估 (評估隨機策略)"""
    V = np.zeros((n, n))
    action_grid = np.random.randint(0, 4, size=(n, n)) # 隨機生成動作
    theta = 1e-4
    
    while True:
        delta = 0
        new_V = np.copy(V)
        for r in range(n):
            for c in range(n):
                s = (r, c)
                if s == st.session_state.end or s in st.session_state.obstacles:
                    continue
                
                # 隨機策略：每個動作機率 0.25
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
                action_grid[r, c] = np.argmax(action_values)
                delta = max(delta, abs(best_value - V[r, c]))
                
        V = new_V
        if delta < theta:
            break
            
    st.session_state.V = V
    st.session_state.action_grid = action_grid
    st.session_state.policy = 'optimal'

# --- UI 介面 ---
init_state()

st.title("網格地圖 RL 導航器 (Gridworld)")

# 側邊欄設定
with st.sidebar:
    st.header("環境設定")
    new_n = st.slider("選擇網格維度 (n)", 5, 9, st.session_state.n)
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

# 狀態提示
phases_zhtw = {
    'set_start': "🟢 請點擊網格設定「起點」",
    'set_end': "🔴 請點擊網格設定「終點」",
    'set_obstacles': f"⬛ 請點擊網格設定「障礙物」 (還需 {st.session_state.n - 2 - len(st.session_state.obstacles)} 個)",
    'ready': "✅ 地圖設定完成！請從側邊欄執行演算法。"
}
st.subheader(phases_zhtw[st.session_state.phase])

# 繪製網格 (使用 Streamlit Columns)
for r in range(st.session_state.n):
    cols = st.columns(st.session_state.n)
    for c in range(st.session_state.n):
        pos = (r, c)
        
        # 決定按鈕顯示的文字與狀態
        cell_text = "⬜"
        is_disabled = False
        
        if pos == st.session_state.start:
            cell_text = "🟢"
        elif pos == st.session_state.end:
            cell_text = "🔴"
        elif pos in st.session_state.obstacles:
            cell_text = "⬛"
        else:
            # 如果有執行演算法，顯示策略方向與 Value
            if st.session_state.policy and st.session_state.V is not None:
                action_sym = ACTIONS[st.session_state.action_grid[r, c]][1]
                val = st.session_state.V[r, c]
                cell_text = f"{action_sym}\n{val:.1f}"
                is_disabled = True # 演算法執行後鎖定網格
                
        # 建立按鈕
        if cols[c].button(cell_text, key=f"btn_{r}_{c}", disabled=is_disabled, use_container_width=True):
            handle_click(r, c)
            st.rerun()

# 顯示當前資訊
if st.session_state.policy:
    st.divider()
    st.markdown(f"**當前顯示狀態：** {'隨機策略 (Random Policy)' if st.session_state.policy == 'random' else '最佳策略 (Optimal Policy)'}")
    st.markdown(f"**$V(s)$ 狀態價值矩陣：**")
    st.dataframe(st.session_state.V)