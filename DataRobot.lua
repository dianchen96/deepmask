-- Data Loader for video datasets: FBMS + VSB + DAVIS
require 'torch'
require'lfs'

local cv = require 'cv'
require 'cv.imgcodecs'

local image = require 'image'
local sys = require 'sys'
local xlua = require 'xlua'    -- xlua provides useful tools, like progress bars
local ffi = require 'ffi'

local DataRobot = torch.class('DataRobot')
<<<<<<< HEAD
=======
-- local dbg = require("debugger")
>>>>>>> dc4cce381c95edf8fd9dc38c9c857e1a0509b110

function DataRobot:__init(config)
    assert(config.data_path, 'Must provide label list file')

    self.dataPath = config.data_path
    self.imgType = config.data_type

    self.posMaskAddr = {}
    self.posImAddr = {}
    self.negImAddr = {}
    

    for model_run in lfs.dir(self.dataPath) do
        local runMaskAddr = {}
        local runImAddr = {}
        if model_run ~= '.' and model_run ~= '..' then
            isNeg = (string.find(model_run, 'neg') ~= nil)
            for file in lfs.dir(self.dataPath .. model_run) do
                if string.find(file,self.imgType) then
                    if string.find(file, 'mask') then
                        dataNum = tonumber(string.match(file, '%d%d%d%d'))
                        runMaskAddr[dataNum] = model_run .. '/' .. file
                    elseif string.find(file, 'img') then
                        dataNum = tonumber(string.match(file, '%d%d%d%d'))
                        runImAddr[dataNum] = model_run .. '/' .. file
                    end
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
    local dataIdx = math.floor(1 + torch.uniform() * #self.posImAddr)
<<<<<<< HEAD
    local img, mask
    if self.imgType == 'tiff' then
        mask = cv.imread{self.dataPath .. self.posMaskAddr[dataIdx], cv.IMREAD_COLOR}
        img = cv.imread{self.dataPath .. self.posImAddr[dataIdx], cv.IMREAD_COLOR}
    else
        mask = image.load(self.dataPath .. self.posMaskAddr[dataIdx])
        img = image.load(self.dataPath .. self.posImAddr[dataIdx])
    end
=======
    -- print("mask address")
    -- print(self.dataPath .. self.posMaskAddr[dataIdx])
    local mask = image.load(self.dataPath .. self.posMaskAddr[dataIdx])
    local img = image.load(self.dataPath .. self.posImAddr[dataIdx])
>>>>>>> dc4cce381c95edf8fd9dc38c9c857e1a0509b110
    return img, mask
end

function DataRobot:randomNegExample()
    local dataIdx = math.floor(1 + torch.uniform() * #self.negImAddr)
<<<<<<< HEAD
    local img
    if self.imgType == 'tiff' then
        img = image.load(self.dataPath .. self.posImAddr[dataIdx])
    else
        img = image.load(self.dataPath .. self.negImAddr[dataIdx])
    end
=======
    local img = image.load(self.dataPath .. self.negImAddr[dataIdx])
>>>>>>> dc4cce381c95edf8fd9dc38c9c857e1a0509b110
    return img
end
