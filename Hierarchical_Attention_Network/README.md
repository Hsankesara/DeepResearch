# Hierarchical Attention Network

Since the uprising of Deep learning and Natural Language Processing, text classification has become one of the most staggering tasks to accomplish. In layman terms, We can say Artificial Intelligence is a field which tries to achieve human-like intelligent models to ease the jobs for all of us. All of us has an astounding proficiency in text classification. But even many sophisticated NLP models are failed to achieve such proficiency. So the question arises is that what we humans do differently? How do we classify text?

First of all we understand words not each and every word but many of them and we can guess even unknown words just by the structure of a sentence. Then we understand the message that those series of words (sentences) conveys. Then from those series of sentences, we understand the meaning of a paragraph or an article. The similar approach is used in Hierarchical Attention model.

## How to use

First install all the necessary dependencies
```bash
bash setup.sh
```

You can test the module using
```bash
python3 run_han.py
``` 
## How to use the module

* To train, test and save your own model first import the HAN module

```python
import HAN
```

* Import your dataset(preferably as a pandas dataframe)
* Import pretrained embedded vector
* Initialize HAN module

```python
han_network = HAN.HAN(text = df.text, labels = df.category, num_categories = total_categories, pretrained_embedded_vector_path = embedded_vector_path, max_features = max_num_of_features, max_senten_len = max_sentence_len, max_senten_num = max_sentence_num , embedding_size = size_of_embedded_vectors)
```
* Tweak hyperparameters using ```set_hyperparametes()``` function of HAN object.
> To know more checkout [run_han.py](run_han.py)

## Dataset courtesy
[News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset)

## Implementation
Go to [this](https://www.kaggle.com/hsankesara/news-classification-using-han) to checkout implementation and functioning of HAN Networks.

## Project Manager

**[Heet Sankesara](https://github.com/Hsankesara)**

[<img src="http://i.imgur.com/0o48UoR.png" width="35" padding="10" margin="10">](https://github.com/Hsankesara/)   [<img src="https://i.imgur.com/0IdggSZ.png" width="35" padding="10" margin="10">](https://www.linkedin.com/in/heet-sankesara-72383a152/)    [<img src="http://i.imgur.com/tXSoThF.png" width="35" padding="10" margin="10">](https://twitter.com/heetsankesara3)   [<img src="https://loading.io/s/icon/vzeour.svg" width="35" padding="10" margin="10">](https://www.kaggle.com/hsankesara)