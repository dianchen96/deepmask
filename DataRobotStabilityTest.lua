require 'torch'
require 'image'

local datarobot = require 'DataRobot'
DataRobot = torch.load('/media/4tb/dian/deepmask_sawyer/robot/dataRobot.t7')

print("test random positive stability")
for i = 1,100000 do
    if i % 1000 == 0 then print("current step %s", i) end
    DataRobot:randomPosExample()
    
end