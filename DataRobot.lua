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
    local img, mask
    if self.imgType == 'tiff' then
        mask = cv.imread{self.dataPath .. self.posMaskAddr[dataIdx], cv.IMREAD_GRAYSCALE}
        img_bgr = cv.imread{self.dataPath .. self.posImAddr[dataIdx], cv.IMREAD_COLOR}
        img_bgr = torch.split(img_bgr, 1, 3)
        img = torch.cat({img_bgr[3], img_bgr[2], img_bgr[1]}, 3):transpose(1,3)
        img = img:double():mul(1./255)
    else
        mask = image.load(self.dataPath .. self.posMaskAddr[dataIdx])
        img = image.load(self.dataPath .. self.posImAddr[dataIdx])
    end
    return img, mask
end

function DataRobot:randomNegExample()
    local dataIdx = math.floor(1 + torch.uniform() * #self.negImAddr)
    local img
    if self.imgType == 'tiff' then
        img_bgr = cv.imread{self.dataPath .. self.negImAddr[dataIdx], cv.IMREAD_COLOR}
        img_bgr = torch.split(img_bgr, 1, 3)
        img = torch.cat({img_bgr[3], img_bgr[2], img_bgr[1]}, 3):transpose(1,3)
        img = img:double():mul(1./255)
    else
        img = image.load(self.dataPath .. self.negImAddr[dataIdx])
    end
    return img
end