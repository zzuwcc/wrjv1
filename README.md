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
