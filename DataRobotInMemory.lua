-- Data Loader for video datasets: FBMS + VSB + DAVIS
require 'torch'
require'lfs'

local cv = require 'cv'
require 'cv.imgcodecs'

local image = require 'image'
local sys = require 'sys'
local xlua = require 'xlua'    -- xlua provides useful tools, like progress bars

local DataRobotInMemory = torch.class('DataRobotInMemory')

function DataRobotInMemory:__init(dataPath, imgType, maxImdim)
    
    self.dataPath = dataPath
    self.imgType = imgType

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
                        runMaskAddr[dataNum] = self.dataPath .. model_run .. '/' .. file
                    elseif string.find(file, 'img') then
                        dataNum = tonumber(string.match(file, '%d%d%d%d'))
                        runImAddr[dataNum] = self.dataPath .. model_run .. '/' .. file
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
        end
    end
    
    -- Assign Tensors
    self.posImages = torch.ByteTensor(#self.posImAddr, 3, maxImdim, maxImdim)
    self.posMasks = torch.ByteTensor(#self.posMaskAddr, maxImdim, maxImdim)
    self.negImages = torch.ByteTensor(#self.negImAddr, 3, maxImdim, maxImdim)
    self.posImages:fill(0)
    self.posMasks:fill(0)
    self.negImages:fill(0)
    
    count = 0
    totalCount = #self.posImAddr + #self.posMaskAddr + #self.negImAddr
    
    print ("Loading Images")
    
    for i=1, #self.posImAddr do
        local img = self:_loadImg(self.posImAddr[i])
        local mask = self:_loadMask(self.posMaskAddr[i])
        self.posImages[{i, {}, {1,mask:size()[1]}, {1,mask:size()[2]}}]:copy(img)
        self.posMasks[{i, {1,mask:size()[1]}, {1,mask:size()[2]}}]:copy(mask)
        count = count + 2
        xlua.progress(count, totalCount)
    end
    
    for i=1, #self.negImAddr do
        local img = self:_loadImg(self.negImAddr[i])
        self.negImages[{i, {}, {1, img:size()[2]}, {1,img:size()[3]}}]:copy(img)
        count = count + 1
        xlua.progress(count, totalCount)
    end
end

function DataRobotInMemory:randomPosExample()
    local dataIdx = math.floor(torch.uniform() * #self.posImAddr)
    local img, mask
    
    img = self.posImages[dataIdx]
    img = img:double():mul(1./255)
    mask = self.posMasks[dataIdx]
    
    return img, mask
end

function DataRobotInMemory:randomNegExample()
    local dataIdx = math.floor(torch.uniform() * #self.posImAddr)
    local img = self.negImages[dataIdx]
    img = img:double():mul(1./255)
    
    return img
end
    
function DataRobotInMemory:_loadImg(imgAddr)
    img_bgr = cv.imread{imgAddr}
    img_bgr = torch.split(img_bgr, 1, 3)
    img = torch.cat({img_bgr[3], img_bgr[2], img_bgr[1]}, 3):transpose(1,3)

    return img
end

function DataRobotInMemory:_loadMask(maskAddr)
    local mask = cv.imread{maskAddr, cv.IMREAD_GRAYSCALE}
    return mask
end

