# DAFSVFND
code for paper: [DAFSVFND : Dual Attention Fusion Network for Fake News Detection on Short Video Platforms]
## Environment
please refer to the file requirements.txt.
## Dataset
We conduct experiments on two datasets: FakeSV and FakeTT. 
### FakeSV
FakeSV:[FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms, AAAI 2023.]
### FakeTT
FakeTT:[FakingRecipe: Detecting Fake News on Short Video Platforms from the Perspective of Creative Process, ACM MM 2024.]
## Data Preprocess
To facilitate reproduction, we provide papers on preprocessing methods that you can use to process the features.Please place these features in the specified location, which can be customized in dataloader.py.<br> 
Bert : [URL of Bert](https://github.com/ymcui/Chinese-BERT-wwm)<br>
MAE : [URL of MAE](https://github.com/facebookresearch/mae)<br>
HuBert : [URL of HuBert](https://github.com/bshall/hubert)

The original dataset can be applied for [FakeSV](https://github.com/ICTMCG/FakeSV) and [FakeTT](https://github.com/ICTMCG/FakingRecipe?tab=readme-ov-file). We have placed it in the data folder.
