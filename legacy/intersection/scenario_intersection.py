"""
scenario_intersection.py
────────────────────────────────────────────────────────
INTERSECTION 데드락 시뮬레이션

레이아웃:  + 형태 교차로 (9 × 9)

    xxxxgxxxx   ← 위쪽 진입로 끝 (목표)
    xxxx.xxxx
    xxxx.xxxx
    xxxx.xxxx
    g.......g   ← 가로 통로 (목표 양끝)
    xxxx.xxxx
    xxxx.xxxx
    xxxx.xxxx
    xxxxgxxxx   ← 아래쪽 진입로 끝 (목표)

    중심 셀: (4,4)

시나리오:
  1) 4대 로봇을 교차로 진입 직전 위치에 배치
       로봇0: (3,4) RIGHT   ← 왼쪽 진입
       로봇1: (4,3) DOWN    ← 위  진입
       로봇2: (5,4) LEFT    ← 오른쪽 진입
       로봇3: (4,5) UP      ← 아래 진입
  2) 모두 FORWARD → 중심 셀(4,4) 쟁탈 → 데드락
  3) 감지 후 해결: 로봇0 후퇴 → 순환 고리 해소
"""

import gymnasium as gym
import rware
from deadlock_core import (
    FORWARD, navigate_to, face_robot,
    DeadlockDetector, resolve_intersection, do_step
)

# ── 레이아웃 ──────────────────────────────────────────────
LAYOUT = """
xxxxgxxxx
xxxx.xxxx
xxxx.xxxx
xxxx.xxxx
g.......g
xxxx.xxxx
xxxx.xxxx
xxxx.xxxx
xxxxgxxxx
"""

# 교차로 진입 직전 위치 & 방향
# (목표 x, 목표 y, 방향)
ENTRY_POSITIONS = [
    (3, 4, 'RIGHT'),   # 로봇0: 왼쪽에서 진입
    (4, 3, 'DOWN'),    # 로봇1: 위에서 진입
    (5, 4, 'LEFT'),    # 로봇2: 오른쪽에서 진입
    (4, 5, 'UP'),      # 로봇3: 아래에서 진입
]


def run():
    print("=" * 60)
    print("SCENARIO 2 : INTERSECTION DEADLOCK")
    print()
    print("  레이아웃 : + 형태 교차로 (9×9)")
    print("  데드락   : 4대 모두 중심(4,4)으로 전진 → 상호 블로킹")
    print("  해결     : 로봇0 후퇴 → 순환 고리 해소")
    print("=" * 60)

    env = gym.make("rware:rware-tiny-4ag-v2", layout=LAYOUT)
    obs, info = env.reset(seed=0)
    n = env.unwrapped.n_agents

    print(f"\n창고 크기 : {env.unwrapped.grid_size}")
    agents = env.unwrapped.agents
    print("초기 위치 :", [(a.x, a.y) for a in agents])

    # ── 설정: 4대를 교차로 진입 직전 위치로 이동 ─────────────
    print("\n[설정] 4대 로봇을 교차로 진입 위치로 이동 중...")
    for idx, (tx, ty, td) in enumerate(ENTRY_POSITIONS):
        success = navigate_to(env, idx, tx, ty, td)
        a = env.unwrapped.agents[idx]
        status = "✓" if success else "△(근사)"
        print(f"  로봇{idx}: ({a.x},{a.y}) {a.dir.name}  {status}")

    pos  = [(a.x, a.y) for a in env.unwrapped.agents]
    dirs = [a.dir.name for a in env.unwrapped.agents]
    print(f"\n설정 완료: 위치={pos}")
    print(f"          방향={dirs}")

    # ── 시뮬레이션 ───────────────────────────────────────────
    print("\n[실행] 모든 로봇 전진 명령 →")
    detector = DeadlockDetector(patience=3)
    resolved = False

    for step in range(60):
        pos  = [(a.x, a.y) for a in env.unwrapped.agents]
        dirs = [a.dir.name for a in env.unwrapped.agents]
        print(f"  step {step:2d} | 위치={pos}")

        # 데드락 감지
        if detector.update(pos) and not resolved:
            print(f"\n  ★ DEADLOCK 감지! (step {step})")
            print(f"     원인: 4대 모두 교차로 진입 불가 (상호 점유)")
            print(f"     해결: 로봇0 후퇴 → 순환 고리 해소\n")
            resolve_intersection(env, verbose=True)
            resolved = True
            detector.reset()
            print(f"\n  ★ 해결 완료! 로봇 재개\n")

        done = do_step(env, [FORWARD] * n)
        if done:
            break

    env.close()
    print("\n[ Intersection 시나리오 완료 ]")
    print()


if __name__ == "__main__":
    run()
