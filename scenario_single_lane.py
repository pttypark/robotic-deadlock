"""
scenario_single_lane.py
────────────────────────────────────────────────────────
SINGLE-LANE 데드락 시뮬레이션

레이아웃:  g...........g   (1행 13열)
           ↑            ↑
         왼쪽 목표   오른쪽 목표

시나리오:
  1) 로봇 2대가 통로 안에 배치됨
  2) 상대적 위치로 방향 결정: 왼쪽 로봇→RIGHT, 오른쪽 로봇→LEFT
  3) 둘 다 FORWARD → 중간에서 충돌 → 데드락 발생
  4) 감지 후 해결: 로봇0 후퇴, 로봇1 통과, 로봇0 재개
"""

import gymnasium as gym
import rware
from deadlock_core import (
    FORWARD, navigate_to, face_robot,
    DeadlockDetector, resolve_single_lane, do_step
)


# ── 레이아웃 ──────────────────────────────────────────────
# 1행짜리 단일 통로 → 로봇이 반드시 이 통로 안에만 존재
# 양 끝 g = 목표 지점 (rware 필수 요소)
LAYOUT = "g...........g"   # 13칸


def run():
    print("=" * 60)
    print("SCENARIO 1 : SINGLE-LANE DEADLOCK")
    print()
    print("  레이아웃 : g...........g  (1줄 13칸 통로)")
    print("  데드락   : 양쪽에서 전진 → 중간에서 충돌")
    print("  해결     : 우선순위 양보 (로봇0 후퇴)")
    print("=" * 60)

    env = gym.make("rware:rware-tiny-2ag-v2", layout=LAYOUT)
    obs, info = env.reset(seed=0)
    n = env.unwrapped.n_agents

    agents = env.unwrapped.agents
    print(f"\n창고 크기 : {env.unwrapped.grid_size}")
    print(f"초기 위치 : 로봇0=({agents[0].x},{agents[0].y})  "
          f"로봇1=({agents[1].x},{agents[1].y})")

    # ── 설정: 상대 위치 기준으로 방향 부여 ──────────────────
    # 왼쪽에 있는 로봇 → RIGHT (상대 쪽으로)
    # 오른쪽에 있는 로봇 → LEFT  (상대 쪽으로)
    left_idx  = 0 if agents[0].x <= agents[1].x else 1
    right_idx = 1 - left_idx

    print(f"\n[설정] 로봇{left_idx}→RIGHT, 로봇{right_idx}→LEFT  "
          f"(서로를 향하도록)")
    face_robot(env, left_idx,  'RIGHT')
    face_robot(env, right_idx, 'LEFT')

    pos = [(a.x, a.y) for a in env.unwrapped.agents]
    dirs = [a.dir.name for a in env.unwrapped.agents]
    print(f"설정 완료 : 위치={pos}  방향={dirs}")

    # ── 시뮬레이션 ───────────────────────────────────────────
    print("\n[실행] 모든 로봇 전진 명령 →")
    detector = DeadlockDetector(patience=3)
    resolved = False

    for step in range(60):
        pos = [(a.x, a.y) for a in env.unwrapped.agents]
        dirs = [a.dir.name for a in env.unwrapped.agents]
        print(f"  step {step:2d} | 위치={pos}  방향={dirs}")

        # 데드락 감지
        if detector.update(pos) and not resolved:
            print(f"\n  ★ DEADLOCK 감지! (step {step})")
            print(f"     원인: 양방향 전진 → 서로 블로킹")
            print(f"     해결: 로봇0 후퇴, 로봇1 통과\n")
            resolve_single_lane(env, verbose=True)
            resolved = True
            detector.reset()
            print(f"\n  ★ 해결 완료! 로봇 재개\n")

        done = do_step(env, [FORWARD] * n)
        if done:
            break

    env.close()
    print("\n[ Single-Lane 시나리오 완료 ]")
    print()


if __name__ == "__main__":
    run()
