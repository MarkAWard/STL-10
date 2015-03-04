require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'mattorch'
-- The type is by default 'double' so I leave it like this now as we never changed it before
-- When using CUDA
--  changes:  require 'cunn', the model, the criterion, input = input:cuda()

------------------------------------ PARAMETERS ----------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-batchSize', 32, 'mini-batch size (1 = pure stochastic)')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-lrDecay', 1e-7, 'learning rate at t=0')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-epochs', 100, 'max number of epochs to run')
cmd:text()
opt = cmd:parse(arg or {})

trainSize     = 4500
valSize       = 500
testSize      = 8000
extraSize     = 100000
channels      = 3
imageHeight   = 96
imageWidth    = 96
outputClasses = 10

torch.setnumthreads( opt.threads )
torch.manualSeed( opt.seed )

if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end 

------------------------------------- READ DATA ----------------------------------------

trainFile = '/scratch/courses/DSGA1008/A2/matlab/train.mat'
testFile = '/scratch/courses/DSGA1008/A2/matlab/test.mat'
extraFile = '/scratch/courses/DSGA1008/A2/matlab/unlabeled.mat'

--trainFile = 'trainA2Matlab.mat'
--testFile  = 'testA2Matlab.mat'
--extraFile = 'unlabeledA2Matlab.mat'

loadedTrain = mattorch.load(trainFile)
loadedTest = mattorch.load(testFile)
--loadedUnlabeled = mattorch.load(extraFile)
allTrainData   = loadedTrain.X:t():reshape(trainSize + valSize, channels, imageHeight, imageWidth)
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
   data   = loadedTest.X:t():reshape(testSize, channels, imageHeight, imageWidth),
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


------------------------------- CREATE SURROGATE CLASS ---------------------------------


------------------------------------ DATA AUGMENTATIONS --------------------------------

--------------------------------- MODEL AND CRITERION -----------------------------------
if opt.type == 'cuda' then
      
   model = nn.Sequential()
   model:add(nn.SpatialZeroPadding(2,2,2,2))
   model:add(nn.SpatialConvolutionMM(3, 23, 7, 7, 2, 2))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(3,3,2,2))
   model:add(nn.Dropout(.5))
   model:add(nn.View(23*23*23))
   model:add(nn.Linear(23*23*23, 50))
   model:add(nn.Linear(50,10))
   model:add(nn.LogSoftMax())

else
   
   model = nn.Sequential()
--   model:add(nn.SpatialZeroPadding(2,2,2,2))
   model:add(nn.SpatialConvolution(3, 23, 7, 7, 2, 2))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(3,3,2,2))
   model:add(nn.Dropout(.5))
   model:add(nn.Reshape(23*22*22))
   model:add(nn.Linear(23*22*22, 50))
   model:add(nn.Linear(50,10))
   model:add(nn.LogSoftMax())

end

criterion = nn.ClassNLLCriterion()

if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------- OPTIMIZATION --------------------------------------

optimState = {
   learningRate = opt.learningRate,
   weightDecay = opt.weightDecay,
   momentum = opt.momentum,
   learningRateDecay = opt.lrDecay
}
optimMethod = optim.sgd

----------------------------------- TRAIN FUNCTION --------------------------------------

function train( epoch )
	classes = {'1','2','3','4','5','6','7','8','9','0'}
	confusion = optim.ConfusionMatrix(classes)

   model:training()    -- set model to training mode (for modules that differ in training and testing, like Dropout)
   shuffle = torch.randperm(trainData:size())   -- shuffle at each epoch
   for t = 1,trainData:size(), opt.batchSize do
      -- create mini batch
      local inputs = {}
      local targets = {}
      for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         local input = trainData.data[shuffle[i]]
         local target = trainData.labels[shuffle[i]]
         if opt.type == 'double' then input = input:double()
         elseif opt.type == 'cuda' then input = input:cuda() end
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
   
   local filename = paths.concat('results', 'model_' .. epoch .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   torch.save(filename, model)
   print(confusion)
   return confusion.totalValid*100
end
--------------------------------- END TRAIN FUNCTION --------------------------------

----------------------------------- VAL FUNCTION --------------------------------------
function val()
   classes = {'1','2','3','4','5','6','7','8','9','0'}
   confusion = optim.ConfusionMatrix(classes)
   model:evaluate()
   for t = 1,valData:size() do
      local input = valData.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = valData.labels[t]
      local pred = model:forward(input)
      confusion:add(pred, target)
   end
   print(confusion)
   return confusion.totalValid * 100
end
--------------------------------- END VAL FUNCTION --------------------------------

------------------------------- MAIN LEARNING FUNCTION ---------------------------------
logger = optim.Logger(paths.concat('results', 'accuracyResults.log'))
logger:add{"EPOCH  TRAIN ACC  VAL ACC"}
for i =1, opt.epochs do
      	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> EPOCH " .. i .. " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<") 
	trainAcc = train(i)
	valAcc   = val()
	logger:add{i .. "," .. trainAcc .. "," ..  valAcc}
end
