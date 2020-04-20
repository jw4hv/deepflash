import modules
def getModel(net_config, loss_config, device):
    net_type = net_config['type']
    if net_type in ['AEFC', 'AECV2D', 'AECV3D']:
        return modules.getAENET(net_config)
    pass