# text-emotion-recogntion

## About

Text Emotion Recognition model to predict emotion from text utterances. Dataset used - IEMOCAP dataset. Model is based on RoBERTa backbone with attention layer to extract contextual information.

## Compatibility
Python - version 3.9 or 3.8
Ran on Google Colab with Tesla T4 GPU
_CudaDeviceProperties(name='Tesla T4', major=7, minor=5, total_memory=15109MB, multi_processor_count=40)

## Dataset
Sentences with Emotion labels- Happy, Sad, Angry, Neutral
Size:-
Train Dataset:  3259
Valid Dataset:  1031
Test Dataset:  1241


## Instructions to run
1. pip install -r requirements.txt
2. python main.py <br>
   <t>[--lr_roberta=\<float\> <t>              (default=2e-6) <br>
    <t>--lr_other=\<float\>        <t>        (default=5e-5)  <br>
    <t>--weight_decay_text=\<float\>  <t>     (default=0.01) <br>
    <t>--weight_decay_other=\<float\>    <t>  (default=1e-3) <br>
    <t>--MAX_EPOCH=\<int\>                 <t>(default=20) <br>
    <t>--MIN_EPOCH=\<int\>                 <t>(default=0) <br>
    <t>]

## Result
Weighted Accuracy: 77.35%
Unweighted Accuracy: 78.46%
F1 Score: 0.7547
