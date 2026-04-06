import gymnasium as gym
import rware  # rware 환경을 gym에 등록시키기 위해 import (이것만 해도 환경 이름 인식됨)

# "작은 창고 + 로봇 4대" 환경 생성
env = gym.make("rware-tiny-4ag-v2")

# 환경 초기화 (seed=42로 고정하면 매번 같은 초기 배치로 시작)
obs, info = env.reset(seed=42)

for step in range(50):
    # 각 에이전트가 취할 행동을 무작위로 샘플링 (로봇 4대 각각 랜덤 행동)
    actions = env.action_space.sample()
    # 한 스텝 진행: 행동 적용 후 관측/보상/종료여부 반환
    obs, rewards, terminated, truncated, info = env.step(actions)

print("실행 성공")
