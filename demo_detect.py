
import time

from MACA.env.radar_reconn_hierarical import RaderReconnHieraricalEnv
from MACA.render.gif_generator import gif_generate

if __name__ == '__main__':
    env = RaderReconnHieraricalEnv({"render": True})

    env.reset()

    done = False
    step = 0
    total_damage = 0
    while not done:
        time.sleep(0.05)

        # actions = {
        #     '1': [0.0], 
        #     '2': [0.0], 
        #     '3': [0.0], 
        #     '4': [0.0], 
        #     '5': [0.0], 
        #     '6': [0.0], 
        # }

        # TODO: 和param中侦察机个数一致
        actions = [[0.0], [0.0], [0.0], [0.0]]
    
        obs, reward, dones, info = env.step(actions)
        env.render(save_pic=True)
        done = dones['__all__']

        step += 1
        total_damage += sum([item[1] for item in info['ally_damage'].items()])
    gif_generate('demo_detect_1.gif')
    print(f'total damage: {total_damage}')

        
        