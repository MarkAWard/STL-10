require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-type', 'cuda', 'type: double | float | cuda')
cmd:option('-batchSize', 32, 'mini-batch size (1 = pure stochastic)')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-lrDecay', .98, 'learning rate at t=0')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-epochs', 400, 'max number of epochs to run')
cmd:text()
opt = cmd:parse(arg or {})
torch.setnumthreads( opt.threads )
torch.setdefaulttensortype('torch.FloatTensor')

--extraSize     = 100000
local channels      = 3
sizeOfPatches = 32
local C = 50 -- number of the initial images we will examine.
local K = 1  -- number of patches from each image
N = 200      -- number of augmentation for each patch

local extraFile = 'un_bin.dat'
local unlabData = torch.load(extraFile).x
------------------------------- CREATE SURROGATE CLASS ---------------------------------
local surrogateSize = C*K*N 
local surrogateData   = torch.zeros( surrogateSize, channels, sizeOfPatches, sizeOfPatches)
local surrogateLabels = torch.zeros( surrogateSize )
-- drawing the indexes of a random samples from the initial unlabeled data
local randomImageIndices = torch.randperm(valData:size())[ {{1,C}} ]
local idx = 1
for i, imageIndex in pairs(randomImageIndices:totable()) do
	local imageToAug = valData.data[imageIndex]
	for k = 1, K do -- we will get 5 random patches from every image
		local randX = math.random( imageHeight - sizeOfPatches )
		local randY = math.random( imageWidth - sizeOfPatches )
		local src = image.crop(imageToAug, randX, randY, randX + sizeOfPatches, randY + sizeOfPatches)
		for j = 1, N do
			surrogateLabels[ idx ] = i -- (i-1)*K+k
			surrogateData[ idx ]   = aug.augment(src)
			idx = idx + 1
		end
	end
end


local percentageForValidation = 10
local startValSurrogate = surrogateSize - (C/percentageForValidation)*K*N 

local surTrainSize   = startValSurrogate
local surValSize     = surrogateSize - startValSurrogate

local surShuffleIndices = torch.randperm(surrogateSize)
local surTrainData   = torch.zeros(surTrainSize, channels, sizeOfPatches, sizeOfPatches)
local surTrainLabels = torch.zeros(surTrainSize)
local surValData     = torch.zeros(surValSize, channels, sizeOfPatches, sizeOfPatches)
local surValLabels   = torch.zeros(surValSize)

for i =1, surTrainSize do
	surTrainData[i]   = surrogateData[ surShuffleIndices[i] ]
	surTrainLabels[i] = surrogateLabels[ surShuffleIndices[i] ]
end
for i=1, surValSize do
	surValData[i]   = surrogateData[ surShuffleIndices[startValSurrogate + i] ]
	surValLabels[i] = surrogateLabels[ surShuffleIndices[startValSurrogate + i] ]
end

trainData = {
   data   = surTrainData,
   labels = surTrainLabels,
   size = function() return surTrainSize end
}
valData = {
   data   = surValData,
   labels = surValLabels,
   size = function() return surValSize end
}





--------------------------------- NORMALIZE SURROGATE TRAINING DATA ----------------------

--[[surrogateData = surrogateData:float()
for i = 1,surrogateSize do
   surrogateData[i] = image.rgb2yuv(surrogateData[i])
end

channelsYUV = {'y','u','v'}
mean = {}
std = {}
-- normalize each channel globally
for i,channel in ipairs(channelsYUV) do
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end
for i,channel in ipairs(channelsYUV) do
	-- Normalize val, test data, using the training means/stds
   valData.data[{ {},i,{},{} }]:add(-mean[i])
   valData.data[{ {},i,{},{} }]:div(std[i])
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end
--]]


--------------------------------- NORMALIZE DATA ---------------------------------------
--[[
trainData.data = trainData.data:float()
valData.data   = valData.data:float()
testData.data  = testData.data:float()
for i = 1,trainSize do
   trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
for i = 1,valSize do
   valData.data[i]   = image.rgb2yuv(valData.data[i])
end
for i = 1,testSize do
   testData.data[i]  = image.rgb2yuv(testData.data[i])
end
channelsYUV = {'y','u','v'}
mean = {}
std = {}

-- normalize each channel globally
for i,channel in ipairs(channelsYUV) do
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end
for i,channel in ipairs(channelsYUV) do
	-- Normalize val, test data, using the training means/stds
   valData.data[{ {},i,{},{} }]:add(-mean[i])
   valData.data[{ {},i,{},{} }]:div(std[i])
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end
-- Normalize all three channels locally
neighborhood = image.gaussian1D(13)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
-- Normalize all channels locally:
for c in ipairs(channelsYUV) do
   for i = 1,trainData:size() do
      trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
   end
   for i = 1,valData:size() do
      valData.data[{ i,{c},{},{} }] = normalization:forward(valData.data[{ i,{c},{},{} }])
   end
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end

print '==> verify statistics'
for i,channel in ipairs(channelsYUV) do
   print('training data, '..channel..'-channel, mean: ' .. trainData.data[{ {},i }]:mean())
   print('training data, '..channel..'-channel, standard deviation: ' .. trainData.data[{ {},i }]:std())
   print('validation data, '..channel..'-channel, mean: ' .. valData.data[{ {},i }]:mean())
   print('validation data, '..channel..'-channel, standard deviation: ' .. valData.data[{ {},i }]:std())
   print('test data, '..channel..'-channel, mean: ' .. testData.data[{ {},i }]:mean())
   print('test data, '..channel..'-channel, standard deviation: ' .. testData.data[{ {},i }]:std())
end
--]]

--[[ REMOVE THE COMMENTS
if opt.type=='cuda' then

require 'cunn'
cutorch.setDevice(3)
cutorch.getDeviceProperties(cutorch.getDevice())

--torch.setdefaulttensortype('torch.CudaTensor')
end
--]]



