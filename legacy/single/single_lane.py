import gymnasium as gym
import numpy as np
import rware

# -------------------------------------------------------
# SINGLE-LANE 데드락 시뮬레이션
#
# 목적: 1칸짜리 좁은 통로에서 두 로봇이 마주쳐 꼼짝 못하는
#       데드락 상황을 재현하고 감지한다.
#
# 레이아웃 구조 (3행 9열):
#   xxxxxxxxx  ← 선반 행 (벽 역할, 로봇 통과 불가)
#   g.......g  ← 단일 통로 (1칸 너비), 양 끝이 목표 지점(g)
#   xxxxxxxxx  ← 선반 행 (벽 역할)
#
# x = 선반 (벽처럼 로봇 이동 차단)
# . = 빈 통로 (이동 가능)
# g = 목표 지점 (선반을 여기로 배달하면 보상)
# -------------------------------------------------------
layout = """
xxxxxxxxx
g.......g
xxxxxxxxx
"""

# 커스텀 레이아웃으로 환경 생성, 로봇 2대
env = gym.make("rware:rware-tiny-2ag-v2", layout=layout)

# seed=42: 항상 동일한 초기 배치로 재현 가능
obs, info = env.reset(seed=42)

# Action.FORWARD = 1 (warehouse.py의 Action Enum 참고)
ACTION_FORWARD = 1
num_robots = env.unwrapped.n_agents  # 실제 로봇 수 확인

print(f"로봇 수: {num_robots}")
print(f"창고 크기: {env.unwrapped.grid_size}")  # (행, 열)
print()

prev_positions = None  # 이전 스텝의 로봇 위치 저장용
deadlock_step = None   # 데드락 발생 스텝 기록용

for step in range(100):
    # 모든 로봇에게 무조건 '전진' 명령
    # → 좁은 통로에서 반대 방향 로봇과 마주치면 둘 다 못 움직임
    actions = [ACTION_FORWARD] * num_robots
    obs, rewards, terminated, truncated, info = env.step(actions)

    # 이번 스텝 후 각 로봇의 (x, y) 위치 수집
    curr_positions = [(agent.x, agent.y) for agent in env.unwrapped.agents]

    print(f"step {step:3d} | 위치: {curr_positions} | 보상: {rewards}")

    # 데드락 판정: 이전 위치와 현재 위치가 동일 = 아무도 못 움직인 것
    if prev_positions is not None and curr_positions == prev_positions:
        deadlock_step = step
        print(f"\n[DEADLOCK] {step}번째 스텝에서 데드락 발생!")
        print(f"원인: 전진 명령을 받았으나 서로 막혀 위치 변화 없음")
        break

    prev_positions = curr_positions  # 다음 스텝 비교를 위해 현재 위치 저장

    # 에피소드 종료 조건 (목표 달성 또는 최대 스텝 초과)
    if terminated or truncated:
        break

if deadlock_step is None:
    print("\n데드락 미발생 (100스텝 내)")
