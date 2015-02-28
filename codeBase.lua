require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'mattorch'
-- The type is by default 'double' so I leave it like this now as we never changed it before
-- When using CUDA
--  changes:  require 'cunn', the model, the criterion, input = input:cuda()
torch.setnumthreads( 8 )
torch.manualSeed(1) -- this was the default

------------------------------------ PARAMETERS ----------------------------------------
trainSize     = 4500
valSize       = 500
testSize     = 800
extraSize     = 100000
channels	  = 3
imageHeight   = 96
imageWidth    = 96
outputClasses = 10

------------------------------------- READ DATA ----------------------------------------
trainFile = 'trainA2Matlab'
testFile  = 'testA2Matlab'
extraFile = 'unlabeledA2Matlab'
loadedTrain = mattorch.load(trainFile)
loadedTest = mattorch.load(testFile)
--loadedUnlabeled = mattorch.load(extraFile)
allTrainData   = loadedTrain.X:t():reshape(trainSize, channels, imageHeight, imageWidth)
allTrainLabels = loadedTrain.y[1]

-- we are going to use the first 4500 indexes of the shuffleIndices as the train set
-- and the 500 last as the validation set
shuffleIndices = torch.randperm(trainSize + valSize)
-- Defining the structures that will hold our data
trainData   = torch.zeros(trainSize, channels, imageHeight, imageWidth)
trainLabels = torch.zeros(trainSize)
valData     = torch.zeros(valSize, channels, imageHeight, imageWidth)
valLabels   = torch.zeros(valSize)

for i =1, trainSize do
	trainData[i]   = allTrainData[ shuffleIndices[i] ]
	trainLabels[i] = allTrainLabels[ shuffleIndices[i] ]
end
-- and now populating the validation data.
for i=1, valSize do
	valData[i]   = allTrainData[ shuffleIndices[i+trainSize] ]
	valLabels[i] = allTrainLabels[ shuffleIndices[i+trainSize] ]
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
   data   = loadedTest.X:t():reshape(testSize, channels, imageHeight, imageWidth)
   labels = loadedTest.y[1],
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
channels = {'y','u','v'}
mean = {}
std = {}

-- normalize each channel globally
for i,channel in ipairs(channels) do
   mean[i] = trainData.data[{ {},i,{},{} }]:mean()
   std[i] = trainData.data[{ {},i,{},{} }]:std()
   trainData.data[{ {},i,{},{} }]:add(-mean[i])
   trainData.data[{ {},i,{},{} }]:div(std[i])
end
for i,channel in ipairs(channels) do
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
for c in ipairs(channels) do
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
for i,channel in ipairs(channels) do
   print('training data, '..channel..'-channel, mean: ' .. trainData.data[{ {},i }]:mean())
   print('training data, '..channel..'-channel, standard deviation: ' .. trainData.data[{ {},i }]:std())
   print('validation data, '..channel..'-channel, mean: ' .. valData.data[{ {},i }]:mean())
   print('validation data, '..channel..'-channel, standard deviation: ' .. valData.data[{ {},i }]:std())
   print('test data, '..channel..'-channel, mean: ' .. testData.data[{ {},i }]:mean())
   print('test data, '..channel..'-channel, standard deviation: ' .. testData.data[{ {},i }]:std())
end


------------------------------- CREATE SURROGATE CLASS ---------------------------------


------------------------------------ DATA AUGMENTATIONS --------------------------------


--------------------------------------- END READ DATA ----------------------------------

------------------------------------ THIS WILL CHANGE ----------------------------------
print '==> define parameters'
noutputs = 10 -- 10-class problem; for us it will be N for the surrogate classes
-- input dimensions
nfeats = 3
width = 32
height = 32
ninputs = nfeats*width*height
-- number of hidden units (for MLP only):
nhiddens = ninputs / 2
-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,128}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)
--------------------------------- END THIS WILL CHANGE ----------------------------------

--------------------------------- MODEL AND CRITERION -----------------------------------
 model = nn.Sequential()
-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))
-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
model:add(nn.Tanh())
model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))
-- stage 3 : standard 2-layer neural network
model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
model:add(nn.Tanh())
model:add(nn.Linear(nstates[3], noutputs))
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
--------------------------------- MODEL AND CRITERION -----------------------------------
optimState = {learningRate = 1e-3, weightDecay = 0, momentum = 0,learningRateDecay = 1e-7}
optimMethod = optim.sgd
batchSize = 128 --  set that to whatever we want



----------------------------------- TRAIN FUNCTION --------------------------------------
function train()
   epoch = epoch or 1   -- epoch tracker
   model:training()    -- set model to training mode (for modules that differ in training and testing, like Dropout)
   shuffle = torch.randperm(trsize)   -- shuffle at each epoch
   for t = 1,trainData:size(), batchSize do
      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         input = input:double()         
         table.insert(inputs, input)
         table.insert(targets, target)
      end

      local feval = function(x) -- create closure to evaluate f(X) and df/dX
      	-- get new parameters
		if x ~= parameters then
			parameters:copy(x)
		end
		gradParameters:zero() -- reset gradients
		local f = 0 -- f is the average of all criterions

		for i = 1,#inputs do -- evaluate function for complete mini batch                          
			local output = model:forward(inputs[i])
			local err = criterion:forward(output, targets[i])
			f = f + err

			local df_do = criterion:backward(output, targets[i])
			model:backward(inputs[i], df_do)
			confusion:add(output, targets[i]) -- update confusion
		end
		gradParameters:div(#inputs) -- normalize gradients and f(X)
		f = f/#inputs
		return f,gradParameters -- return f and df/dX
	  end
      optimMethod(feval, parameters, optimState)
      
   end
   print( confusion.totalValid*100 ) -- tracking accuracy
   -- for next epoch
   confusion:zero()
   epoch = epoch + 1
end
--------------------------------- END TRAIN FUNCTION --------------------------------

----------------------------------- VAL FUNCTION --------------------------------------
function val()
   model:evaluate()
   for t = 1,valData:size() do
      local input = valData.data[t]
      input = input:double()
      local target = valData.labels[t]
      local pred = model:forward(input)
      confusion:add(pred, target)
   end
   print(confusion.totalValid * 100)   
   -- next iteration:
   confusion:zero()
   return valAccuracy
end
--------------------------------- END VAL FUNCTION --------------------------------

