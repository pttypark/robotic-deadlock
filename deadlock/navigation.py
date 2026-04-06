"""
navigation.py
─────────────────────────────────────────
로봇 이동 유틸리티

- 방향 계산 (회전 시퀀스, 반대 방향)
- 단일 스텝 이동
- 목표 위치까지 자동 이동 (navigate_to)
"""

# ── 액션 상수 (warehouse.py Action Enum 기준) ─────────────
NOOP       = 0
FORWARD    = 1
TURN_LEFT  = 2
TURN_RIGHT = 3

# 방향 순서: 시계방향 (warehouse.py wraplist 기준)
WRAPLIST = ['UP', 'RIGHT', 'DOWN', 'LEFT']


# ── 방향 유틸리티 ─────────────────────────────────────────

def turn_seq(curr_d, target_d):
    """curr_d → target_d 최단 회전 시퀀스 반환"""
    ci = WRAPLIST.index(curr_d)
    ti = WRAPLIST.index(target_d)
    cw  = (ti - ci) % 4
    ccw = (ci - ti) % 4
    return ([TURN_RIGHT] * cw) if cw <= ccw else ([TURN_LEFT] * ccw)

def rev_dir(d):
    """반대 방향 반환"""
    return {'UP':'DOWN', 'DOWN':'UP', 'LEFT':'RIGHT', 'RIGHT':'LEFT'}[d]


# ── 환경 조작 ─────────────────────────────────────────────

def do_step(env, actions):
    """env.step 실행 후 종료 여부 반환"""
    _, _, terminated, truncated, _ = env.step(actions)
    return terminated or truncated

def face_robot(env, idx, target_dir):
    """로봇 idx 를 target_dir 방향으로 회전 (나머지 NOOP)"""
    n = env.unwrapped.n_agents
    agent = env.unwrapped.agents[idx]
    for action in turn_seq(agent.dir.name, target_dir):
        acts = [NOOP] * n
        acts[idx] = action
        do_step(env, acts)

def _move_forward(env, idx, n):
    """로봇 idx 한 칸 전진 (나머지 NOOP)"""
    acts = [NOOP] * n
    acts[idx] = FORWARD
    do_step(env, acts)


def navigate_to(env, idx, tx, ty, td, max_steps=300):
    """
    로봇 idx 를 (tx, ty) 로 이동 후 td 방향으로 정렬.
    다른 로봇은 NOOP 유지.
    막히면 수직 우회 시도.
    """
    n = env.unwrapped.n_agents

    for _ in range(max_steps):
        a = env.unwrapped.agents[idx]
        if a.x == tx and a.y == ty and a.dir.name == td:
            return True

        if a.x != tx or a.y != ty:
            d = ('RIGHT' if a.x < tx else 'LEFT') if a.x != tx else \
                ('DOWN'  if a.y < ty else 'UP')

            px, py = a.x, a.y
            face_robot(env, idx, d)
            _move_forward(env, idx, n)

            a = env.unwrapped.agents[idx]
            if a.x == px and a.y == py:   # 막힌 경우 수직 우회
                face_robot(env, idx, 'DOWN' if a.y == 0 else 'UP')
                _move_forward(env, idx, n)
        else:
            face_robot(env, idx, td)

    return False
