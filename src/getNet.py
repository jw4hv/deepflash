from modules import AENet
def getNet(net_config):
    net_type = net_config['type']
    if net_type in ['AEFC', 'AECV2D', 'AECV3D']:
        return AENet.getAENET(net_config)
    else:
        raise ValueError(f'Unsupported Net Type: {net_type}')
    pass