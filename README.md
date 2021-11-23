# MultiTask-LSTM-HAR
This is the implementation of the paper "Multitask LSTM Model for Human Activity Recognition and Intensity Estimation Using Wearable Sensor Data" (IEEE IoT Journal), 2020.

## Dataset

The dataset splits can be downloaded [here](https://github.com/ACANETS/HAR-UML20), please create `data/` folder, download the csv files and place them in the `data/` folder. 

## Requirements

- python 3.5.8 or higher
- matplotlib
- tensorflow-gpu 1.6.0 or higher

## Create and setup the virtual Environment

```shell
python3 -m venv ./env
```

```shell
source ./env/bin/activate
```

```shell
pip install -r requirements.txt
```

## Training and Testing

When the virtual environment is activated:

```shell
python3 wearable-main.py --function clfOnly --saveModel --testSubject 10
```

Setting the hyperparameters is optional. For details:

```shell
python3 wearable-main.py --help
```

## References

If this repository was useful for your research, please cite.

```
@ARTICLE{UML-HAR20,
  author={Barut, Onur and Zhou, Li and Luo, Yan},
  journal={IEEE Internet of Things Journal}, 
  title={Multitask LSTM Model for Human Activity Recognition and Intensity Estimation Using Wearable Sensor Data}, 
  year={2020},
  volume={7},
  number={9},
  pages={8760-8768},
  doi={10.1109/JIOT.2020.2996578}}
```