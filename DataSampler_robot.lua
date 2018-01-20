require 'torch'
require 'image'

-- data format:
-- img 448 x 448, the center 224 is canonical form

local tds = require 'tds'
local datarobot = require 'datarobot'
local DataSampler_robot = torch.class('DataSampler_robot')

--------------------------------------------------------------------------------
-- function: init
function DataSampler_robot:__init(config,split)
  assert(split == 'train' or split == 'val')

  -- dian api, already have all file path saved
  self.dian = torch.load('path to saved t7 object') -- .pos() randomly return 1 img 1 msk, .neg() randomly return 1 img.

  -- mean/std computed from random subset of ImageNet training images
  self.mean, self.std = {0.485, 0.456, 0.406}, {0.229, 0.224, 0.225}

  -- class members
  self.split = split
  self.scaleCorrectFactor = log2(224/160)
  self.iSz = config.iSz --current value is 160
  self.objSz = math.ceil(config.iSz*128/224) -- object's max axis length is: 160 * (4/7) = 91.5
  self.wSz = config.iSz + 32 -- 192 the real feedin size
  self.gSz = config.gSz --mask size 112 no need to change
  self.scale = config.scale
  self.shift = config.shift
  self.negJetteringMin = config.negJetteringMin
  self.negJetteringMax = config.negJetteringMax
  
  
  if split == 'train' then self.__size  = config.maxload*config.batch
  elseif split == 'val' then self.__size = config.testmaxload*config.batch end

  -- self.scales stores all bad range
  if config.hfreq > 0 then
    self.scales = {} -- scale range for score sampling
    for scale = -2,-0.5,0.25 do table.insert(self.scales,scale) end
    for scale = 0.5,2,.25 do table.insert(self.scales,scale) end
  end

  collectgarbage()
end
local function log2(x) return math.log(x)/math.log(2) end

--------------------------------------------------------------------------------
-- function: get size of epoch
function DataSampler_robot:size()
  return self.__size
end

--------------------------------------------------------------------------------
-- function: get a sample, no change needed
function DataSampler_robot:get(headSampling)
  local input,label
  if headSampling == 1 then -- sample masks
    input, label = self:maskSampling()
  else -- sample score
    input,label = self:scoreSampling()
  end

  if torch.uniform() > .5 then
    input = image.hflip(input)
    if headSampling == 1 then label = image.hflip(label) end
  end

  -- normalize input
  for i=1,3 do input:narrow(1,i,1):add(-self.mean[i]):div(self.std[i]) end

  return input,label
end

--------------------------------------------------------------------------------
-- function: mask sampling, return positive data
-- return image, mask pair w.t. jettering and scaling if no_scaling_jettering = 0
-- else return image, mask pair in canonical form
function DataSampler_robot:positiveSampling(no_scaling_jettering)
  local iSz,wSz,gSz = self.iSz,self.wSz,self.gSz

  -- get image
  local inp, mask = self.dian:pos()
  local scale = torch.uniform(-self.scale, self.scale) -- default value: self.scale = 0.25
  if no_scaling_jettering == 1 then
    scale = 1
  end
  scale = scale + self.scaleCorrectFactor
  --jittering
  local xc, yc, w, h = 224, 224, 224, 224
  xc = xc + torch.uniform(-self.shift,self.shift)*2^s
  yc = yc + torch.uniform(-self.shift,self.shift)*2^s
  if no_scaling_jettering == 1 then
    xc, yc = 224, 224
  end

  w, h = wSz*2^scale, wSz*2^scale
  local bbox = {xc - w/2, yc - h/2, w, h}
  
  -- crop & resize image
  inp = self:cropTensor(inp, bbox, 0.5)
  inp = image.scale(inp, wSz, wSz)
  
  --crop & resize mask
  lbl = self:cropTensor(mask, bbox, 0)
  lbl = image.scale(lbl, gSz, gSz) -- make sure that mask is already binary

  return inp, lbl
end
--------------------------------------------------------------------------------
-- function: sample negative data from negative data
function DataSampler_robot:negativeSampling()
  local iSz,wSz,gSz = self.iSz,self.wSz,self.gSz
  local inp = self.dian:neg()
  local scale = torch.uniform(-1, 1)
  local xc, yc = 224, 224
  local side = 224*2^scale
  local bbox = {xc-side/2, yc-side/2, side, side}
  return self:cropTensor(inp, bbox, 0.5)
end

--------------------------------------------------------------------------------
-- function: return negative data from positive data
function DataSampler_robot:negativeFromPositiveSampling()
  local iSz,wSz,gSz = self.iSz,self.wSz,self.gSz
  -- get image
  local inp, mask = self.dian:pos()
  local scale = 0
  local shiftlow = 0
  local shiftupper = self.negJetteringMax -- may have some padding issue (Here I use a larger negative scale range)
  local scale1 = torch.uniform(-self.scale, self.scale)
  local index = math.random(1,#self.scales)
  local scale2 = self.scales[index]
  if torch.uniform() > .3 then
    scale = scale2 --bad scale, arbitrary jettering
  else 
    scale = scale1 --god scale, bad jettering
    shiftlow = self.negJetteringMin
  end
  scale = scale + self.scaleCorrectFactor
  local sign = torch.random(0,1)*2 - 1
  side = 2^scale * wSz
end
--------------------------------------------------------------------------------
-- function: score head sampler
local imgPad = torch.Tensor()
function DataSampler_robot:scoreSampling(cat,imgId)
  local idx,bb
  repeat
    idx = torch.random(1,self.nImages)
    bb = self.bbStruct[idx]
  until #bb.scales ~= 0

  local imgId = self.imgIds[idx]
  local imgName = self.coco:loadImgs(imgId)[1].file_name
  local pathImg = string.format('%s/%s2014/%s',self.datadir,self.split,imgName)
  local img = image.load(pathImg,3)
  local h,w = img:size(2),img:size(3)

  -- sample central pixel of BB to be used
  local x,y,scale
  local lbl = torch.Tensor(1)
  if torch.uniform() > .5 then
    x,y,scale = self:posSamplingBB(bb)
    lbl:fill(1)
  else
    x,y,scale = self:negSamplingBB(bb,w,h)
    lbl:fill(-1)
  end

  local s = 2^-scale
  x,y  = math.min(math.max(x*s,1),w), math.min(math.max(y*s,1),h)
  local isz = math.max(self.wSz*s,10)
  local bw =isz/2

  --pad/crop/rescale
  imgPad:resize(3,h+2*bw,w+2*bw):fill(.5)
  imgPad:narrow(2,bw+1,h):narrow(3,bw+1,w):copy(img)
  local inp = imgPad:narrow(2,y,isz):narrow(3,x,isz)
  inp = image.scale(inp,self.wSz,self.wSz)

  return inp,lbl
end

--------------------------------------------------------------------------------
-- function: crop bbox b from inp tensor
function DataSampler_robot:cropTensor(inp, b, pad)
  pad = pad or 0
  b[1], b[2] = torch.round(b[1])+1, torch.round(b[2])+1 -- 0 to 1 index
  b[3], b[4] = torch.round(b[3]), torch.round(b[4])

  local out, h, w, ind
  if #inp:size() == 3 then
    ind, out = 2, torch.Tensor(inp:size(1), b[3], b[4]):fill(pad)
  elseif #inp:size() == 2 then
    ind, out = 1, torch.Tensor(b[3], b[4]):fill(pad)
  end
  h, w = inp:size(ind), inp:size(ind+1)

  local xo1,yo1,xo2,yo2 = b[1],b[2],b[3]+b[1]-1,b[4]+b[2]-1
  local xc1,yc1,xc2,yc2 = 1,1,b[3],b[4]

  -- compute box on binary mask inp and cropped mask out
  if b[1] < 1 then xo1=1; xc1=1+(1-b[1]) end
  if b[2] < 1 then yo1=1; yc1=1+(1-b[2]) end
  if b[1]+b[3]-1 > w then xo2=w; xc2=xc2-(b[1]+b[3]-1-w) end
  if b[2]+b[4]-1 > h then yo2=h; yc2=yc2-(b[2]+b[4]-1-h) end
  local xo, yo, wo, ho = xo1, yo1, xo2-xo1+1, yo2-yo1+1
  local xc, yc, wc, hc = xc1, yc1, xc2-xc1+1, yc2-yc1+1
  if yc+hc-1 > out:size(ind)   then hc = out:size(ind  )-yc+1 end
  if xc+wc-1 > out:size(ind+1) then wc = out:size(ind+1)-xc+1 end
  if yo+ho-1 > inp:size(ind)   then ho = inp:size(ind  )-yo+1 end
  if xo+wo-1 > inp:size(ind+1) then wo = inp:size(ind+1)-xo+1 end
  out:narrow(ind,yc,hc); out:narrow(ind+1,xc,wc)
  inp:narrow(ind,yo,ho); inp:narrow(ind+1,xo,wo)
  out:narrow(ind,yc,hc):narrow(ind+1,xc,wc):copy(
  inp:narrow(ind,yo,ho):narrow(ind+1,xo,wo))

  return out
end

--------------------------------------------------------------------------------
-- function: crop bbox from mask
function DataSampler_robot:cropMask(ann, bbox, h, w, sz)
  local mask = torch.FloatTensor(sz,sz)
  local seg = ann.segmentation

  local scale = sz / bbox[3]
  local polS = {}
  for m, segm in pairs(seg) do
    polS[m] = torch.DoubleTensor():resizeAs(segm):copy(segm); polS[m]:mul(scale)
  end
  local bboxS = {}
  for m = 1,#bbox do bboxS[m] = bbox[m]*scale end

  local Rs = self.maskApi.frPoly(polS, h*scale, w*scale)
  local mo = self.maskApi.decode(Rs)
  local mc = self:cropTensor(mo, bboxS)
  mask:copy(image.scale(mc,sz,sz):gt(0.5))

  return mask
end

--------------------------------------------------------------------------------
-- function: jitter bbox
function DataSampler_robot:jitterBox(box)
  local x, y, w, h = box[1], box[2], box[3], box[4]
  local xc, yc = x+w/2, y+h/2
  local maxDim = math.max(w,h)
  local scale = log2(maxDim/self.objSz)
  local s = scale + torch.uniform(-self.scale,self.scale)
  xc = xc + torch.uniform(-self.shift,self.shift)*2^s
  yc = yc + torch.uniform(-self.shift,self.shift)*2^s
  w, h = self.wSz*2^s, self.wSz*2^s
  return {xc-w/2, yc-h/2,w,h}
end

--------------------------------------------------------------------------------
--function: posSampling: do positive sampling
function DataSampler_robot:posSamplingBB(bb)
  local r = math.random(1,#bb.scales)
  local scale = bb.scales[r]
  r=torch.random(1,#bb[scale])
  local x,y = bb[scale][r][1], bb[scale][r][2]
  return x,y,scale
end

--------------------------------------------------------------------------------
--function: negSampling: do negative sampling
function DataSampler_robot:negSamplingBB(bb,w0,h0)
  local x,y,scale
  local negSample,c = false,0
  while not negSample and c < 100 do
    local r = math.random(1,#self.scales)
    scale = self.scales[r]
    x,y = math.random(1,w0*2^scale),math.random(1,h0*2^scale)
    negSample = true
    for s = -10,10 do
      local ss = scale+s*self.scale
      if bb[ss] then
        for _,c in pairs(bb[ss]) do
          local dist = math.sqrt(math.pow(x-c[1],2)+math.pow(y-c[2],2))
          if dist < 3*self.shift then
            negSample = false
            break
          end
        end
      end
      if negSample == false then break end
    end
    c=c+1
  end
   return x,y,scale
end

return DataSampler_robot
