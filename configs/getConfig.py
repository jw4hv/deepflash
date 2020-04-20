
def getConfig(name):
    if name == 'default':
        ae_config = {
                'data':{
                    'training': ['./data/mri_1.mhd', './data/mri_4.mhd'],
                    'valid':['./data/mri_1.mhd'],
                    'test':['./data/mri_1.mhd'],
                    'preprocessing':{

                        'crop':{
                            'method': 'sliding',
                            'patch_shape': (32,32,32),
                            'strides':(4,4,4)
                            },
                        'augmentation': 
                            None,
                                    }
                        },
                'net':
                    {
                        'type': 'AECV3D',                        
                        'paras': {
                            'structure': 'default',
                            'batchNorm': False
                                }
                        # 'paras': {
                        #     'structure': 'decreasing',
                        #     'batchNorm': True,
                        #     'root_feature_num':16
                        #         }
                     },
                'training':{
                        'epochs_num': 10,
                        'batch_size': 1,
                        'learning_rate':1e-5,
                        'report_per_epochs':20,
                        'training_check': False,
                        'valid_check': {
                                'mode':'save_img',
                                'index':0,   # could be int or'random' 
                                'slice_axis':2,  # for 3D images. Could be 0,1,2
                                'slice_index': 'middle' # for 3D images. Could be int or 'middle'
                                  },
                        'save_trained_model': True},
                'groundtruth':{
                        None},
                'loss':{
                        'name': 'MSE',
                        'para': None
                        }
        }
    elif name == 'debug':
        ae_config = getConfig('default')
        ae_config['data']['augmentation'] = None
        ae_config['data']['crop'] = None
        ae_config['net']['paras']['name'] = 'debug'
        ae_config['training']['epochs_num'] = 10
        ae_config['training']['report_per_epochs'] = 1

    elif name == 'deepflash':
        ae_config = getConfig('default')
        ae_config['data']['augmentation'] = None
        ae_config['net']['type'] = 'DF'
        ae_config['net']['paras']['structure']  = 'deepflash'
        ae_config['training']['epochs_num'] = 500
        ae_config['training']['batch_size'] = 32
        ae_config['training']['learning_rate'] = 7e-4
        ae_config['loss']['name'] = 'MSE'
        ae_config['training']['report_per_epochs'] = 1
    return ae_config

def getConfigGroup(name):
    configs = []
    if name == 'debug':
        configName = 'debug'
        config = getConfig(configName)
        config['idxInGroup'] = 1
        configs.append(config)

        config = getConfig(configName)
        config['idxInGroup'] = 2
        config['training']['batch_size'] = 100
        configs.append(config)
    elif name == 'general-test':
        # Baseline config
        config0Name = 'default'
        config0 = getConfig(config0Name)
        config0['idxInGroup'] = 1
        config0['name'] = 'baseline'
        config0['net']['paras']['structure'] = 'decreasing'
        config0['net']['paras']['batchNorm'] = False
        config0['net']['paras']['decreasing_layer_num'] = 3
        config0['net']['paras']['root_feature_num'] = 16
        configs.append(config0)

        # BatchNorm
        configBN = config0        
        configBN['idxInGroup'] = 2
        configBN['name'] = 'batchNorm'
        config0['net']['paras']['batchNorm'] = True
        configs.append(configBN)

        # Smaller stride
        configSS = config0
        configSS['idxInGroup'] = 3
        configSS['name'] = 'Smaller Stride'
        configSS['data']['preprocessing']['crop']['strides'] = (1,1,1)
        configs.append(configSS)


        # More or less layers
        configFL = config0
        configFL['idxInGroup'] = 5
        configFL['name'] = 'Fewer Layers'
        configFL['net']['paras']['decreasing_layer_num'] = 2
        configs.append(configFL)

        configML = config0
        configML['idxInGroup'] = 4
        configML['name'] = 'More Layers'
        configML['net']['paras']['decreasing_layer_num'] = 4
        configs.append(configML)
        
    return configs