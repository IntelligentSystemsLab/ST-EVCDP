## Spatio-temporal EVCDP (Shenzhen)

This is a real-world dataset for spatio-temporal electric vehicle (EV) charging demand prediction. If it is helpful to your research, please cite our paper:

>Qu, H., Kuang, H., Li, J., & You, L. (2023). A physics-informed and attention-based graph learning approach for regional electric vehicle charging demand prediction. IEEE Transactions on Intellgent Transportation Systems. [Paper in IEEE Explore](https://ieeexplore.ieee.org/document/10539613) [Paper in arXiv](https://arxiv.org/abs/2309.05259)

```shell
@ARTICLE{10539613,
  author={Qu, Haohao and Kuang, Haoxuan and Wang, Qiuxuan and Li, Jun and You, Linlin},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={A Physics-Informed and Attention-Based Graph Learning Approach for Regional Electric Vehicle Charging Demand Prediction}, 
  year={2024},
  pages={1-14},
  doi={10.1109/TITS.2024.3401850}}
```

Author: Haohao Qu (haohao.qu@connect.polyu.hk)

## Updates
* **July 2, 2024: We are excited to announce the release of the early access version of ST-EVCDP-v2! You can download the data from [Google Drive Link](https://drive.google.com/drive/folders/1sqOUEpMh8VMiJhrT-MOn5OsB1-KirUVq?usp=drive_link).**
* May 30, 2024: The ST-EVCDP-V2 dataset is still being compiled... We will make every effort to release it as quickly as we can.
* May 14, 2024:  Our paper has been accepted by [IEEE T-ITS](https://ieeexplore.ieee.org/document/10539613)!
* ST-EVCDP-Version2 is coming soon. The data will span from September 2022 to September 2023, with granularity to charging stations, including coordinates, numbers of chargers, occupancy, and price. Other information will also be gradually released after being desensitized for academic research purposes.
* May 12, 2024: We uploaded the data of weather conditions in the studied areas, namely `SZweather20220619-20220718.csv` and `SZweather_Header.txt`.
* March 15, 2024: We uploaded the data of charging duration and volume in the studied areas.

## Data Description

### ST-EVCDP
The data used in this study is drawn from a publicly available mobile application, which provides the real-time availability of charging piles (i.e., idle or not). Within **Shenzhen**, China, a total of 18,061 public charging piles are covered during the studied period from 19 June to 18 July 2022 (30 days) with a minimum interval of 5 minutes and `8640 timestamps`. As shown in Figure 1, the city is constructed into a graph-structure data with `247 nodes` (traffic zones) and `1006 edges` (adjacent relationships).

![avatar](figs/map.png) Figure 1. Spatial distribution of the 18,061 public EV charging piles in ST-EVCDP.

Besides, the pricing schemes for the studied charging piles are also collected. Among the 247 traffic zones, 57 of them (enclosed in red lines) deploy time-based pricing schemes, while others use fixed ones. More statistical details are illustrated in the following table.

![avatar](figs/statistics.png)

### ST-EVCDP-v2
Expanding on the foundation of ST-EVCDP, we have gathered an extensive dataset called ST-EVCDP-v2, specifically tailored for EV-related research. This dataset covers a timeframe of **one year**, spanning from September 2022 to September 2023, which includes comprehensive information such as coordinates, charging occupancy, duration, volume, and price for a total of 1,682 public charging stations with 24,798 public charging piles. Notably, it provides detailed information on charging stations, with a granularity that allows analysis at the charging station level. And its temporal interval is one hour. You can download the data from [Google Drive Link](https://drive.google.com/drive/folders/1sqOUEpMh8VMiJhrT-MOn5OsB1-KirUVq?usp=drive_link).

![avatar](figs/map_v2.png) Figure 2. Spatial distribution of the 24,798 public EV charging piles in ST-EVCDP-v2.

## Files
### ST-EVCDP
* `adj.csv`: The adjacent matrix of studied areas, 1 indicates the two traffic zones are neighboring, vice versa.
* `distance.csv`: Distances between nodes.
* `information.csv`: Several basis information about the data, including pile capacity, longitude, latitude, whether or not located in the central business district (1:yes, 0:no), and whether or not on a time-based pricing scheme (1:yes, 0:no).
* `occupancy.csv`: The real-time EV charging occupancy in studied areas.
* `duration.csv`: The real-time EV charging duration in studied areas, i.e., the sum of charging time for all charging piles, unit in hour.
* `volume.csv`: The real-time EV charging volume in studied areas, i.e., the total power consumption of all charging piles, unit in kWh.
* `price.csv`: The real-time EV charging pricing in studied areas.
* `time.csv`: The timestamps of studied period.
* `Shenzhen.qgz`: The QGIS map file of Shenzhen city.

### ST-EVCDP-v2
* `inf.csv`: Important information of the charging stations, including coordinates and charging capacities.
* `count.csv`: Hourly EV charging occupancy (busy count) in certain stations.
* `duration.csv`: Hourly EV charging duration in specific stations (Unit: hour)
* `volume.csv`: Hourly EV charging volume in specific stations (Unit: kWh)
* `e_price.csv`: Electricity price for specific stations.
* `s_price.csv`: Service price for specific stations.

## Enviroment Requirement
```shell
pip install -r requirements.txt
```

## An simple example to run Spatio-temporal Prediction on the dataset

We developed a physics-informed and attention-based approach for spatio-temporal EV charging demand prediction, named **PAG**. Expect that, some representative methods are included, e.g., LSTM, and GCN-LSTM, GAT-LSTM. You can train and test the proposed model through the following procedures:

1. Choose your model in line 45 of `main.py` or use the default model (PAG-) by skipping this procedure.
2. Run `main.py` via Pycharm, etc. or change your ROOT_PATH and command:

```shell
cd [path] && python main.py
```

## Extend:
* If you want to run your own models on the datasets we offer, you should go to `models.py` and replace the model in `main.py`.


More updates will be posed in the near future! Thank you for your interest.
