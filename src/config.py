import os
import argparse

from src.utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            'wmsa': self.__WMSA
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelname)
        dataset_name = str.lower(args.dataset)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        root_dataset_dir = '../Datasets'
        tmp = {
            'iemocap':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir, 'SIMS/unaligned_39.pkl'),
                    'seq_lens': (39, 400, 55), # (text, audio, video)
                    'feature_dims': (768, 33, 709), # (text, audio, video)
                    'train_samples': 1368,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss'
                }
            }
        }
        return tmp      
 
    def __WMSA(self):
        tmp = {
            'commonParas':{
                'need_normalized': False,
                'use_bert': True,
                'use_finetune': True,
            },
            # dataset
            'datasetParas':{
                'iemocap': {
                    'early_stop': 4,
                    # training/validation/test parameters
                    'batch_size': 16,
                    # 'learning_rate_bert': 2e-5,
                    # 'learning_rate_other': 5e-5,
                    # 'weight_decay_bert': 0.001,
                    # 'weight_decay_other': 0.001,

                    # multi-head attention
                    # 'num_heads': 5,
                    # 'num_layers': 8,
                    # 'embed_dim_a': 5,
                    # 'embed_dim_v': 20,
                    # 'attn_dropout_a': 0.0,
                    # 'attn_dropout_v' :0.0,
                    # 'relu_dropout': 0.1,
                    # 'res_dropout': 0.1,
                    # 'embed_dropout': 0.25,

                    # 'random_switch': 0.8,
                    # network parameters
                    # 'audio_in': 1875,#5,#1875,
                    # 'video_in': 10000,#20,#10000,

                    'text_out': 1024,
                    'attn_heads': 4,
                    'audio_out': 400,
                    'video_out': 400,
                    'post_fusion_dim': 128,
                    'post_layer_dim': 512,
                    'post_dim': 128,
                    'output_dim': 4,
                    # 'post_fusion_dropout': 0.1,
                    # 'post_text_dropout': 0.1,
                    # 'post_audio_dropout': 0.5,#0.05,
                    # 'post_video_dropout': 0.5,#0.1,
                    # 'post_hybrid_dropout': 0.1,
                    #
                    'weight_k': 2,
                    # res
                    'H': 3.0
                },
            },
        }
        return tmp

    def get_config(self):
        return self.args