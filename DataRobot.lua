-- Data Loader for video datasets: FBMS + VSB + DAVIS
require 'torch'
require'lfs'

local image = require 'image'
local sys = require 'sys'
local xlua = require 'xlua'    -- xlua provides useful tools, like progress bars
local ffi = require 'ffi'

local DataRobot = torch.class('DataRobot')

function DataRobot:__init(config)
    assert(config.data_path, 'Must provide label list file')
    self.dataPath = config.data_path

    self.posMaskAddr = {}
    self.posImAddr = {}
    self.negImAddr = {}

    for model_run in lfs.dir(self.dataPath) do
        local runMaskAddr = {}
        local runImAddr = {}
        if model_run ~= '.' and model_run ~= '..' then
            isNeg = (string.find(model_run, 'neg') ~= nil)
            for file in lfs.dir(self.dataPath .. model_run) do
                if string.find(file, 'mask') then
                    dataNum = tonumber(string.match(file, '%d%d%d%d'))
                    runMaskAddr[dataNum] = model_run .. '/' .. file
                elseif string.find(file, 'img') then
                    dataNum = tonumber(string.match(file, '%d%d%d%d'))
                    runImAddr[dataNum] = model_run .. '/' .. file
                end
            end

            assert (isNeg or #runImAddr == #runMaskAddr, model_run .. ' is corrupted! Img and mask num does not match')
            
            if isNeg then
                for i=0, #runImAddr do
                    imAddr = runImAddr[i]
                    self.negImAddr[#self.negImAddr + 1] = imAddr
                end
            else
                for i=0, #runImAddr do
                    maskAddr = runMaskAddr[i]
                    imAddr = runImAddr[i]
                    self.posMaskAddr[#self.posMaskAddr + 1] = maskAddr
                    self.posImAddr[#self.posImAddr + 1] = imAddr
                end
            end

            print ("Finished processing " .. model_run)
        end
    end

end

function DataRobot:randomPosExample()
    local dataIdx = math.ceil(1 + torch.uniform() * #self.posImAddr)
    local mask = image.load(self.dataPath .. self.posMaskAddr[dataIdx])
    local img = image.load(self.dataPath .. self.posImAddr[dataIdx])
    return img, mask
end

function DataRobot:randomNegExample()
    local dataIdx = math.ceil(1 + torch.uniform() * #self.negImAddr)
    print(self.negImAddr[dataIdx])
    local img = image.load(self.dataPath .. self.negImAddr[dataIdx])
    return img
end