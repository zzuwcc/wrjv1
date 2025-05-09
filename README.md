# MACA-2D

MACA-2D is a multi-agent air combat secnario, based on [MACA](https://github.com/CETC-TFAI/MaCA).

## Quick Start

#### Install

```shell
pip install -r requirements.txt
```

#### Train & Test

```shell
git clone https://github.com/xwqianbei/wrjv1.git
cd wrjv1
```

- run the demo_detect
```shell
python demo_detect.py
```

- train the ippo on detect env
```shell
python ./MACA/algorithm/ippo/Runner_detect.py
```

- test the ippo model on detect env
```shell
python ./MACA/algorithm/ippo/TestPolicy_detect.py
```
#### Modify map configuration
`./MACA/utils`下有侦察环境的地图配置，以`param_map_{number}`命名
在`./MACA\env\radar_reconn_hierarical.py`的`args_map = get_args("param_map_1.yaml")`可以切换相应的地图配置
