## Unet

## Dataset
Images for segmentation of optical coherence tomography images with diabetic macular edema.

* You can download dataset  from [here](https://drive.google.com/open?id=1yTwLNnJloMC9t-8cfhnp0fHZpwisGUv0)

* Download and unzip the data on Unet directory

## How to use the module

First install all the necessary dependencies
```bash
pip3 install -r requirements.txt
```

* Download the dataset and save it in Unet directory
* To train, test and save your own model first import the Unet module

```python
import Unet
```

```python
"""
width_out : width of the output image
height_out : height of the output image
width_in : width of the input image
height_in : height of the input image
"""
unet = Unet.Unet(inchannels, outchannnels)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)
outputs = outputs.permute(0, 2, 3, 1)
m = outputs.shape[0]
outputs = outputs.resize(m*width_out*height_out, 2)
labels = labels.resize(m*width_out*height_out)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

> To know more checkout [run_unet.py](run_unet.py)

## Implementation
Go to [this](https://www.kaggle.com/hsankesara/unet-image-segmentation) to checkout implementation and functioning of Unet Networks.

## Project Manager

**[Heet Sankesara](https://github.com/Hsankesara)**

[<img src="http://i.imgur.com/0o48UoR.png" width="35" padding="10" margin="10">](https://github.com/Hsankesara/)   [<img src="https://i.imgur.com/0IdggSZ.png" width="35" padding="10" margin="10">](https://www.linkedin.com/in/heet-sankesara-72383a152/)    [<img src="http://i.imgur.com/tXSoThF.png" width="35" padding="10" margin="10">](https://twitter.com/heetsankesara3)   [<img src="https://loading.io/s/icon/vzeour.svg" width="35" padding="10" margin="10">](https://www.kaggle.com/hsankesara)