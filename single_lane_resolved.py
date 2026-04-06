import gymnasium as gym
import numpy as np
import rware

# -------------------------------------------------------
# SINGLE-LANE 데드락 해결 시뮬레이션
#
# 전략: 우선순위 기반 양보 (Priority-based Yielding)
#   - 데드락 감지 시 로봇 0번이 뒤로 물러남
#   - 로봇 1번은 계속 전진
#   - 로봇 0번이 충분히 물러나면 로봇 1번이 통과 가능
#
# 실행 흐름:
#   Phase 1 (일반 주행): 모든 로봇 전진 → 데드락 발생
#   Phase 2 (해결 중):   로봇 0 후진, 로봇 1 전진 → 교행
#   Phase 3 (해결 완료): 교행 후 정상 주행 재개
# -------------------------------------------------------

layout = """
xxxxxxxxx
g.......g
xxxxxxxxx
"""

# seed=19: 로봇0=(2,1) RIGHT, 로봇1=(6,1) DOWN
# 로봇1을 RIGHT 한 번 돌리면 LEFT가 되어 서로 마주보는 상태 완성
GOOD_SEED = 19

env = gym.make("rware:rware-tiny-2ag-v2", layout=layout)
obs, info = env.reset(seed=GOOD_SEED)

# Action Enum (warehouse.py 기준)
NOOP          = 0
FORWARD       = 1
TURN_LEFT     = 2
TURN_RIGHT    = 3

num_robots = env.unwrapped.n_agents

print(f"로봇 수: {num_robots}")
print(f"창고 크기: {env.unwrapped.grid_size}")

# 초기 방향 출력
for i, agent in enumerate(env.unwrapped.agents):
    print(f"  로봇{i}: ({agent.x},{agent.y}) 방향={agent.dir.name}")
print()

# ── 준비 단계: 로봇1(DOWN)을 RIGHT 한 번 돌려서 LEFT로 만들기 ──
# seed=19 기준: 로봇0=RIGHT, 로봇1=DOWN → RIGHT 한 번 → LEFT
# 그러면 로봇0(RIGHT)과 로봇1(LEFT)이 서로 마주보는 상태 완성
print("[준비] 로봇1 방향 조정 중...")
obs, rewards, terminated, truncated, info = env.step([NOOP, TURN_RIGHT])
for i, agent in enumerate(env.unwrapped.agents):
    print(f"  로봇{i}: ({agent.x},{agent.y}) 방향={agent.dir.name}")
print()

prev_positions = None
phase = "일반주행"       # 현재 실행 단계
yield_steps = 0          # 양보(후진) 진행 스텝 수
YIELD_DURATION = 4       # 로봇 0번이 몇 스텝 동안 후진할지

# 후진 시퀀스: 180도 회전(LEFT 2번) 후 전진
# LEFT → LEFT → FORWARD → FORWARD → FORWARD → FORWARD
yield_sequence = [TURN_LEFT, TURN_LEFT, FORWARD, FORWARD, FORWARD, FORWARD]

for step in range(100):

    if phase == "일반주행":
        # 모든 로봇 전진 → 데드락 유도
        actions = [FORWARD] * num_robots

    elif phase == "해결중":
        # 로봇 0: 후진 시퀀스 수행 (뒤로 물러남)
        # 로봇 1: 계속 전진
        robot0_action = yield_sequence[yield_steps] if yield_steps < len(yield_sequence) else FORWARD
        actions = [robot0_action, FORWARD]
        yield_steps += 1

        # 후진 시퀀스 완료 → 해결 완료
        if yield_steps >= len(yield_sequence):
            phase = "해결완료"

    elif phase == "해결완료":
        # 양보 완료 후 모든 로봇 정상 전진 재개
        actions = [FORWARD] * num_robots

    obs, rewards, terminated, truncated, info = env.step(actions)
    curr_positions = [(agent.x, agent.y) for agent in env.unwrapped.agents]

    print(f"[{phase}] step {step:3d} | 위치: {curr_positions} | 보상: {rewards}")

    # 데드락 감지 → 해결 단계로 전환
    if phase == "일반주행" and prev_positions is not None and curr_positions == prev_positions:
        print(f"\n  >>> [DEADLOCK 감지] → 해결 전략 실행: 로봇 0번 양보\n")
        phase = "해결중"
        yield_steps = 0

    prev_positions = curr_positions

    if terminated or truncated:
        print("\n에피소드 종료")
        break

print("\n시뮬레이션 완료")
