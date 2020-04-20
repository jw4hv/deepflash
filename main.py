from src.runDeepFLASH import runDeepFLASH
from configs.getConfig import getConfig

resultPath = '../result'
configName = 'default'
config = getConfig(configName)
runExp(config, configName, resultPath)