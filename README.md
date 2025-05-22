# MACA-2D

MACA-2D is a multi-agent air combat secnario, based on [MACA](https://github.com/CETC-TFAI/MaCA).

## Quick Start

### git clone and install

```shell
git clone https://github.com/xwqianbei/wrjv1.git
cd wrjv1
```

```shell
pip install -r requirements.txt
```

### Train & Test

#### Run the demo of maps
- run the demo_detect
```shell
python demo_detect.py
```


#### Run scripts to train and test
- run the script to train and test
```shell
sh ./MACA/scripts/dz_train.sh
```

#### Run python to train

- set the map_name:
  - `dz_easy` `dz_medium` `dz_hard`
  - `zc_easy` `zc_medium` `zc_hard`

- train the ippo on battle env
```shell
python ./MACA/algorithm/ippo/Runner.py --total_steps 200 --number 0 --map_name "dz_easy"
```


- test the ippo model on battle env
```shell
python ./MACA/algorithm/ippo/Runner.py --total_steps 200 --number 0 --map_name "dz_easy"
```

- train the ippo on the detect env
```shell
python ./MACA/algorithm/ippo/Runner_detect.py --total_steps 200 --number 0 --map_name "zc_easy"
```

- test the ippo on the detect env
```shell
python ./MACA/algorithm/ippo/TestPolicy_detect.py --number 0 --map_name "zc_easy"
```

#### Modify map configuration
`./MACA/utils`下有侦察环境的地图配置，以`param_map_{number}`命名
在`./MACA\env\radar_reconn_hierarical.py`的`args_map = get_args("param_map_1.yaml")`可以切换相应的地图配置
