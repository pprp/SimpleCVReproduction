import gym
# 创建环境
env = gym.make('CartPole-v0')
# 重置环境状态
env.reset()
for _ in range(1000):
    # 弹出窗口，渲染一帧
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    if done:
        print("reward:", reward)
        env.reset()
# 关闭环境，清理内存    
env.close()

