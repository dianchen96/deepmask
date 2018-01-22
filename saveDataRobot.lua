require 'DataRobot'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('save object of DataRobot class')
cmd:text()
cmd:option('-data_type', 'jpg', 'image type')
cmd:option('-data_path', '/media/4tb/dian/window_seg_dataset_lua/', 'path of masks and images')
cmd:option('-save_dir', '/media/4tb/dian/deepmask_sawyer/robot/')

local config = cmd:parse(arg)

local dataRobot = DataRobot(config)
local save_path = config.save_dir .. config.data_type .. 'DataRobot.t7'
torch.save(save_path, dataRobot)
print ("Saved to " .. save_path)