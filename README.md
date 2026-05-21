# RBY1 Whole-Body Controller

`rby1-wbc` is a codebase for whole-body controller on the rainbow RBY1 robot.

## Setup
Follow `control/README.md` to build the RBY1 realtime controller pybind module.

## Examples
Configure gRPC address in config/wbc.yaml, localhost:50051 to run in simulation, 192.168.30.1:50051 to run in the real world.
```bash
python scripts/rby1_wbc_gui.py
# drag the end-effectors in GUI
```

Kinematics only:
```bash
python scripts/rby1_wbik_gui.py
```

## Running with a high-level visuomotor policy
Checkout [HoMMI](https://github.com/xxm19/hommi)

## 📜 Citation
If you find this repo useful for your research, please cite:

```console
@article{xu2026hommi,
	title={HoMMI: Learning Whole-Body Mobile Manipulation from Human Demonstrations},
	author={Xu, Xiaomeng and Park, Jisang and Zhang, Han and Cousineau, Eric and Bhat, Aditya and Barreiros, Jose and Wang, Dian and Song, Shuran},
	journal={arXiv preprint arXiv:2603.03243},
	year={2026}
	}
```
