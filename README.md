# Visible-Xray Cross-Modality Package Re-Identification
The code will be released.
This repository is Pytorch code for our proposed Baseline method for Cross-Modality Package Re-Identification. 

### Dataset.
The dataset RX01 is privacy sensitive and is currently only supported for academic research. If you need it, please contact us. sxchan@zjut.edu.cn.

### Training.
  Train a model by
  ```bash
python train.py 
```

  - `--dataset`: which dataset "RX01" or "sysu".

### Result.

The results may have some fluctuation, and might be better by finetuning the hyper-parameters.


|Datasets    | Rank@1   | mAP     |
| --------   | -----    |  -----  |
|#RX01      | ~ 48.0%  | ~ 59.0% |
|#SYSU-MM01[1]  | ~ 68.54%  | ~ 66.53% |


### Citation

Our code is based on [Yukang Zhang](https://github.com/ZYK100/Towards-a-Unified-Middle-Modality-Learning-for-Visible-Infrared-Person-Re-Identification) [2]. 

###  References.

[1] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[2] Zhang, Yukang, et al. "Towards a unified middle modality learning for visible-infrared person re-identification." Proceedings of the 29th ACM International Conference on Multimedia. 2021.




