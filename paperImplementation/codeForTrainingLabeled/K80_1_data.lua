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

-- x is an image
-- w is the window of the patch
function patch_finder(x,w)
	model=nn.SpatialAveragePooling(w,w,1,1)
	--x=image.rotate(trainData.data[k],-1.5707963268)
	x_grad=image.rgb2y(image.convolve(x, image.laplacian(8)))
	x_grad=torch.abs(x_grad)
	 
	output=model:forward(x_grad)
	max_val=torch.max(output)
	for i=1, (#output)[2] do
	    for j=1, (#output)[2] do
		holder=output[{{1},{i},{j}}]:reshape(1)
		if holder[1]==max_val then
		    tmp={i,j} end
		end
	end
	return(x[{{},{tmp[1],tmp[1]+w},{tmp[2],tmp[2]+w}}])
end


trainSize     = 4500
valSize       = 500
testSize      = 8000
extraSize     = 100000
channels      = 3
imageHeight   = 32
imageWidth    = 32
outputClasses = 10
C = 5000 

torch.setdefaulttensortype('torch.FloatTensor')
trainFile = 'tr_bin.dat'
testFile = 'ts_bin.dat'
--extraFile = 'un_bin.dat'

loadedTrain=torch.load(trainFile)
loadedTest =torch.load(testFile)
loadedTrain.x = loadedTrain.x:float()
loadedTest.x  = loadedTest.x:float()

allTrainData=loadedTrain.x
allTrainLabels = loadedTrain.y

shuffleIndices = torch.randperm(trainSize + valSize)
trainData   = torch.zeros(trainSize, channels, imageHeight, imageWidth)
trainLabels = torch.zeros(trainSize)
valData     = torch.zeros(valSize, channels, imageHeight, imageWidth)
valLabels   = torch.zeros(valSize)
testData     = torch.zeros(testSize, channels, imageHeight, imageWidth)

for i =1, trainSize do
	trainData[i]   = patch_finder(allTrainData[ shuffleIndices[i] ], 31)
	--trainData[i]   = image.crop(allTrainData[ shuffleIndices[i] ], 0,0,32,32)
	trainLabels[i] = allTrainLabels[ shuffleIndices[i] ]
end
-- and now populating the validation data.
for i=1, valSize do
	valData[i]   = patch_finder(allTrainData[ shuffleIndices[i+trainSize] ], 31)
	--valData[i]   = image.crop(allTrainData[ shuffleIndices[i+trainSize] ], 0,0,32,32)
	valLabels[i] = allTrainLabels[ shuffleIndices[i+trainSize] ]
end
for i=1, testSize do
	testData[i]   = patch_finder(loadedTest.x[i], 31)
	--testData[i]   = image.crop(loadedTest.x[i], 0,0,32,32)
end

trainData = {
   data   = trainData,
   labels = trainLabels,
   size = function() return trainSize end
}
valData = {
   data   = valData,
   labels = valLabels,
   size = function() return valSize end
}
testData = {
   data   = testData,
   labels = loadedTest.y,
   size = function() return testSize end
}

--------------------------------- NORMALIZE DATA ---------------------------------------
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

if opt.type=='cuda' then

require 'cunn'
cutorch.setDevice(3)
cutorch.getDeviceProperties(cutorch.getDevice())

--torch.setdefaulttensortype('torch.CudaTensor')
end



