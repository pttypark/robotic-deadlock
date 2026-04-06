import gymnasium as gym
import rware

layout = """
xxxxxxxxx
g.......g
xxxxxxxxx
"""

# 씨드별 로봇 초기 상태 출력
for seed in range(30):
    e = gym.make("rware:rware-tiny-2ag-v2", layout=layout)
    e.reset(seed=seed)
    a0, a1 = e.unwrapped.agents
    print(f"seed={seed:3d} | 로봇0: ({a0.x},{a0.y}) {a0.dir.name:5s} | 로봇1: ({a1.x},{a1.y}) {a1.dir.name:5s} | x차이={abs(a0.x-a1.x)}")
    e.close()
