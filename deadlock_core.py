"""
deadlock_core.py
─────────────────────────────────────────────
공통 유틸리티, 데드락 감지, 데드락 해결 모듈

이 파일은 시나리오 파일들(scenario_*.py)에서 import해서 씁니다.
"""

import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────
# 액션 상수  (warehouse.py Action Enum 기준)
# ──────────────────────────────────────────
NOOP        = 0
FORWARD     = 1
TURN_LEFT   = 2
TURN_RIGHT  = 3

# 방향 순서: 시계방향 (warehouse.py wraplist 기준)
WRAPLIST = ['UP', 'RIGHT', 'DOWN', 'LEFT']

# ──────────────────────────────────────────
# 방향 유틸리티
# ──────────────────────────────────────────

def turn_seq(curr_d, target_d):
    """curr_d → target_d 최단 회전 시퀀스 반환"""
    ci = WRAPLIST.index(curr_d)
    ti = WRAPLIST.index(target_d)
    cw  = (ti - ci) % 4   # 오른쪽 회전 횟수
    ccw = (ci - ti) % 4   # 왼쪽 회전 횟수
    return ([TURN_RIGHT] * cw) if cw <= ccw else ([TURN_LEFT] * ccw)

def rev_dir(d):
    """반대 방향 반환"""
    return {'UP':'DOWN', 'DOWN':'UP', 'LEFT':'RIGHT', 'RIGHT':'LEFT'}[d]


# ──────────────────────────────────────────
# 환경 조작 유틸리티
# ──────────────────────────────────────────

def do_step(env, actions):
    """env.step 실행, 종료 여부 반환"""
    _, _, terminated, truncated, _ = env.step(actions)
    return terminated or truncated

def face_robot(env, idx, target_dir):
    """로봇 idx 를 target_dir 방향으로 회전 (나머지는 NOOP)"""
    n = env.unwrapped.n_agents
    agent = env.unwrapped.agents[idx]
    for action in turn_seq(agent.dir.name, target_dir):
        acts = [NOOP] * n
        acts[idx] = action
        do_step(env, acts)

def navigate_to(env, idx, tx, ty, td, max_steps=300):
    """
    로봇 idx 를 (tx, ty) 위치로 이동 후 td 방향으로 정렬.
    다른 로봇은 NOOP 상태 유지.
    막히면 수직 우회 시도.
    """
    n = env.unwrapped.n_agents

    for _ in range(max_steps):
        a = env.unwrapped.agents[idx]
        if a.x == tx and a.y == ty and a.dir.name == td:
            return True  # 도착

        if a.x != tx or a.y != ty:
            # 목표 방향 결정 (x 우선)
            if a.x != tx:
                d = 'RIGHT' if a.x < tx else 'LEFT'
            else:
                d = 'DOWN'  if a.y < ty else 'UP'

            # 이동 전 위치 기록
            px, py = a.x, a.y
            face_robot(env, idx, d)
            acts = [NOOP] * n
            acts[idx] = FORWARD
            do_step(env, acts)

            a = env.unwrapped.agents[idx]
            # 막혀서 못 움직인 경우 → 수직 우회
            if a.x == px and a.y == py:
                detour = 'DOWN' if a.y == 0 else 'UP'
                face_robot(env, idx, detour)
                acts = [NOOP] * n
                acts[idx] = FORWARD
                do_step(env, acts)
        else:
            # 위치 도착, 방향만 조정
            face_robot(env, idx, td)

    return False  # 실패


# ──────────────────────────────────────────
# 데드락 감지
# ──────────────────────────────────────────

class DeadlockDetector:
    """
    N 스텝 연속으로 모든 로봇 위치가 동일하면 데드락으로 판정.
    기본값: patience=3 (3스텝 연속 미동 시 감지)
    """
    def __init__(self, patience=3):
        self.patience = patience
        self.history  = []

    def update(self, positions):
        """
        positions: [(x0,y0), (x1,y1), ...]
        returns:   True = 데드락 감지
        """
        self.history.append(tuple(map(tuple, positions)))
        if len(self.history) < self.patience + 1:
            return False
        recent = self.history[-(self.patience + 1):]
        return all(r == recent[0] for r in recent[1:])

    def reset(self):
        self.history = []


# ──────────────────────────────────────────
# 데드락 해결: Single-Lane
# ──────────────────────────────────────────

def resolve_single_lane(env, verbose=True):
    """
    전략: 우선순위 기반 양보 (Priority-based Yielding)
      - 로봇 0 (낮은 우선순위): 180도 회전 → 후퇴 5칸 → 다시 원래 방향
      - 로봇 1 (높은 우선순위): 계속 전진
    효과: 로봇 1이 통과한 뒤 로봇 0이 원래 방향으로 재개
    """
    agents  = env.unwrapped.agents
    r0_dir  = agents[0].dir.name

    # 로봇0 시퀀스: 뒤돌기 → 5칸 후퇴 → 다시 앞으로
    r0_seq = (
        turn_seq(r0_dir, rev_dir(r0_dir))        # 180도 회전
        + [FORWARD] * 5                           # 후퇴
        + turn_seq(rev_dir(r0_dir), r0_dir)       # 다시 원래 방향
    )
    # 로봇1: 전체 구간 동안 계속 전진
    r1_seq = [FORWARD] * len(r0_seq)

    for r0a, r1a in zip(r0_seq, r1_seq):
        do_step(env, [r0a, r1a])
        if verbose:
            pos = [(a.x, a.y) for a in env.unwrapped.agents]
            dirs = [a.dir.name for a in env.unwrapped.agents]
            print(f"    [해결중] 위치={pos}  방향={dirs}")


# ──────────────────────────────────────────
# 데드락 해결: Intersection
# ──────────────────────────────────────────

def resolve_intersection(env, verbose=True):
    """
    전략: 우선순위 가장 낮은 로봇(0번)이 후퇴 → 순환 고리 끊기
      - 로봇 0: 180도 회전 → 2칸 후퇴 (공간 확보)
      - 나머지: 로봇 0이 회전하는 동안 NOOP, 이후 전진
    효과: 로봇 0이 물러나면 막혔던 사슬이 풀려 모두 이동 가능
    """
    n       = env.unwrapped.n_agents
    agents  = env.unwrapped.agents
    r0_dir  = agents[0].dir.name

    n_turns  = len(turn_seq(r0_dir, rev_dir(r0_dir)))
    r0_seq   = turn_seq(r0_dir, rev_dir(r0_dir)) + [FORWARD] * 2
    # 나머지: 로봇0 회전 구간 동안 NOOP, 이후 FORWARD
    other_seq = [NOOP] * n_turns + [FORWARD] * 2

    for i in range(len(r0_seq)):
        acts = [r0_seq[i]] + [other_seq[i]] * (n - 1)
        do_step(env, acts)
        if verbose:
            pos  = [(a.x, a.y) for a in env.unwrapped.agents]
            dirs = [a.dir.name for a in env.unwrapped.agents]
            print(f"    [해결중] 위치={pos}  방향={dirs}")
