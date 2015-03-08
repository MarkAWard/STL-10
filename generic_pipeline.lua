require 'torch'       -- torch
require 'image'       -- for color transforms
require 'nn'          -- provides a normalization operator
require 'optim'
local data = require 'data_preprocess'
local mod  = require 'model'
local crit = require 'criterion'
local aug  = require 'augmentations'

------------------------------------ PARAMETERS ----------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-machine', 'k80', 'k80 or hpc')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 8, 'number of threads')
cmd:option('-device', 3, 'gpu device to use')
cmd:option('-type', 'cuda', 'type: double | float | cuda')

cmd:option('-model', 'cuda', 'name of the model to use')

cmd:option('-loss', 'nll', 'loss function to use')
cmd:option('-batchSize', 32, 'mini-batch size (1 = pure stochastic)')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-lrDecay', 1e-7, 'learning rate at t=0')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-epochs', 100, 'max number of epochs to run')

cmd:option('-train', 'train.mat', 'filepath for training data')
cmd:option('-test', 'test.mat', 'filepath for test data')
cmd:option('-extra', 'extra.mat', 'filepath for extra data')

cmd:option('-trainSize', 4500, 'training set size')
cmd:option('-valSize', 500, 'validation set size')
cmd:option('-testSize', 8000, 'testing set size')
cmd:option('-extraSize', 0, 'extra data set size')

cmd:option('-augment', true, 'augment and increase training dataset')
cmd:option('-augSize', 200, 'number of new samples to create per image')
cmd:option('-flip', 0.5, 'probability for transformation')
cmd:option('-translate', 0.5, 'probability for transformation')
cmd:option('-scale', 0.5, 'probability for transformation')
cmd:option('-rotate', 0.5, 'probability for transformation')
cmd:option('-contrast', 0.5, 'probability for transformation')
cmd:option('-color', 0.5, 'probability for transformation')

cmd:option('-results', 'results', 'name of directory to put results in')

cmd:text()
opt = cmd:parse(arg or {})

-- problem specific image size
channels = 3
imageHeight = 96
imageWidth = 96

-- set environment and defaults
torch.setnumthreads( opt.threads )
torch.manualSeed( opt.seed )
torch.setdefaulttensortype('torch.FloatTensor')
if opt.type == 'cuda' then
	print('==> switching to CUDA')
	require 'cunn'
	-- IS THIS ONLY FOR K80????
	if opt.machine == 'k80' then
		cutorch.setDevice(opt.device)
	else
		cutorch.setDevice(1)
	end
	cutorch.getDeviceProperties(cutorch.getDevice())
end 

-- set filepaths
trainFile = opt.train
testFile = opt.test
extraFile = opt.extra

print('==> loading in data files')
-- load in the data using the machine specific function
local loader = torch.load
if opt.machine == 'hpc' then
	require 'mattorch'    -- loading .mat files
	loader = mattorch.load
end
if opt.trainSize ~= 0 then
	print('    training data...')
	loadedTrain = loader(trainFile)
end
if opt.testSize ~= 0 then
	print('    test data...')
	loadedTest = loader(testFile)
end
if opt.extraSize ~= 0 then
	print('    extra data...')
	loadedExtra = loader(extraFile)
end

-- machines load different formatted datasets
-- $$$$$$$$$ TODO add in extra data $$$$$$$$$$$$$
print('==> formatting data')
if opt.machine == 'hpc' then
	if opt.trainSize ~= 0 then
		allTrainData   = loadedTrain.X:t():reshape(opt.trainSize + opt.valSize, channels, imageHeight, imageWidth)
		allTrainLabels = loadedTrain.y[1]
	end
	if opt.testSize ~= 0 then
		allTestData    = loadedTest.X:t():reshape(testSize, channels, imageHeight, imageWidth)
		allTestLabels  = loadedTest.y[1]
	end
end
-- $$$$$$$$$ TODO add in extra data $$$$$$$$$$$$$
if opt.machine == 'k80' then
	if opt.trainSize ~= 0 then
		allTrainData   = loadedTrain.x
		allTrainLabels = loadedTrain.y
	end
	if opt.testSize ~= 0 then
		allTestData    = loadedTest.x
		allTestLabels  = loadedTest.y
	end
end

-- Defining the structures that will hold our data
-- $$$$$$$$$ TODO add in extra data $$$$$$$$$$$$$
if opt.trainSize ~= 0 then
	if opt.augment then
		number_of_images = opt.trainSize * (opt.augSize + 1)
	else
		number_of_images = opt.trainSize
	end
	trainData   = torch.zeros(number_of_images, channels, imageHeight, imageWidth)
	trainLabels = torch.zeros(number_of_images)
end
if opt.valSize ~= 0 then
	valData     = torch.zeros(opt.valSize, channels, imageHeight, imageWidth)
	valLabels   = torch.zeros(opt.valSize)
end

-- shuffle dataset 
shuffleIndices = torch.randperm(opt.trainSize + opt.valSize)
for i =1, opt.trainSize do
	trainData[i]   = allTrainData[ shuffleIndices[i] ]
	trainLabels[i] = allTrainLabels[ shuffleIndices[i] ]
end
-- and now populating the validation data.
for i=1, opt.valSize do
	valData[i]   = allTrainData[ shuffleIndices[i+opt.trainSize] ]
	valLabels[i] = allTrainLabels[ shuffleIndices[i+opt.trainSize] ]
end

-- create more data
print('==> creating augmented data')
idx = 1
if opt.augment then
	-- iterate through each image
	for i = 1, opt.trainSize do
		local imageToAug = trainData[i]
		local imageLabel = trainLabels[i]
		-- perform augSize augmentations on each image
		for j = 1, opt.augSize do
			trainData[opt.trainSize + idx]   = aug.augment(imageToAug, opt)
			trainLabels[opt.trainSize + idx] = imageLabel
			idx = idx + 1
		end
	end
end

-- create final data objects
-- $$$$$$$$$ TODO add in extra data $$$$$$$$$$$$$
if opt.trainSize ~= 0 then
	trainData = {
	   data   = trainData,
	   labels = trainLabels,
	   size = function() return number_of_images end
	}
end
if opt.valSize ~= 0 then
	valData = {
	   data   = valData,
	   labels = valLabels,
	   size = function() return opt.valSize end
	}
end
if opt.testSize ~= 0 then
	testData = {
	   data   = allTestData,
	   labels = allTestLabels,
	   size = function() return opt.testSize end
	}
end

local mean = {}
local std = {}

-- normalize data and convert to yuv format
print('==> normalizing data')
mean, std = data.normalize_data(trainData, valData, testData)

print('==> setting model and criterion')
model = mod.select_model(opt)
criterion = crit.select_criterion(opt)

if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

print('Size of training data: ' .. trainData:size())
print(model)
print(criterion)



----------------------------------- TRAIN FUNCTION --------------------------------------

function train( epoch )
	classes = {'1','2','3','4','5','6','7','8','9','0'}
	local confusion = optim.ConfusionMatrix(classes)
	model:training() -- set model to training mode (for modules that differ in training and testing, like Dropout)
	-- Shuffling the training data   
	shuffle = torch.randperm(trainData:size())
	shuffed_tr_data=torch.zeros(trainData:size(),channels,imageHeight,imageWidth)
	shuffed_tr_targets=torch.zeros(trainData:size())	
	for t = 1, trainData:size() do
		shuffed_tr_data[t]=trainData.data[shuffle[t]]
		shuffed_tr_targets[t]=trainData.labels[shuffle[t]]
	end
	
	-- batch training to exploit CUDA optimizations
	parameters,gradParameters = model:getParameters()
	local clr = 0.1
	local no_wrong=0
	for t = 1,trainData:size(), opt.batchSize do
		local inputs  = shuffed_tr_data[{{t, math.min(t+opt.batchSize-1, trainData:size())}}]
		local targets = shuffed_tr_targets[{{t, math.min(t+opt.batchSize-1, trainData:size())}}]
		if opt.type=='cuda' then 
			inputs=inputs:cuda()
			targets=targets:cuda()
		end
		gradParameters:zero()
		
		local output = model:forward(inputs)
		local f = criterion:forward(output, targets)
		local trash, argmax = output:max(2)
	  	if opt.type=='cuda' then  argmax=argmax:cuda() else argmax=argmax:float() end
	  	
	  	no_wrong = no_wrong + torch.ne(argmax, targets):sum()
	  	model:backward(inputs, criterion:backward(output, targets))

		--clr = opt.learningRate * (0.5 ^ math.floor(epoch / opt.lrDecay))

		clr = opt.learningRate
		
		
		parameters:add(-clr, gradParameters)
		
		argmax=argmax:reshape((#inputs)[1])
		confusion:batchAdd(argmax, targets)
   end

   local filename = paths.concat(opt.results, 'model_' .. epoch .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   torch.save(filename, model)
   --print(confusion)
   return no_wrong/(trainData:size())   
end
--------------------------------- END TRAIN FUNCTION --------------------------------

-------------------------------- EVALUATE FUNCTION --------------------------------------
function evaluate( modelPath, dataset, writeToFile)
	modelToEval = torch.load(modelPath)
	local f
	if writeToFile then
	   local outputFile = paths.concat(opt.results, 'output.csv')
	   f = io.open(outputFile, "w")
	   f:write("Id,Category\n")
	end
	
	modelToEval:evaluate()
	local no_wrong = 0
	for t = 1,dataset:size(), opt.batchSize do
		local inputs  = dataset.data[{{t, math.min(t+opt.batchSize-1, dataset:size())}}]
		local targets = dataset.labels[{{t, math.min(t+opt.batchSize-1, dataset:size())}}]
		if opt.type == 'cuda' then 
			inputs  = inputs:cuda() 
			targets = targets:cuda()
    	end
    	local output = modelToEval:forward(inputs)
    	local trash, argmax = output:max(2)
    	no_wrong = no_wrong + torch.ne(argmax, targets):sum()
    	
    	if writeToFile then
    		for idx = 1, opt.batchSize do
    			f:write( t+idx-1 .. " , " .. argmax[idx][1] .. "\n") 
    		end
    	end 

    end
	if writeToFile then f:close() end
    return no_wrong/(dataset:size())
end




logger = optim.Logger(paths.concat(opt.results, 'errorResults.log'))
logger:add{"EPOCH    TRAIN ERROR    VAL ERROR"}

valErrorEpochPair = {1.1,-1}
for epoch =1, opt.epochs do
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> EPOCH " .. epoch .. " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<") 
	trainErr = train( epoch )
	print(trainErr)
	valErr   = evaluate( paths.concat(opt.results,'model_'.. epoch ..'.net'), valData, false)
	if valErr < valErrorEpochPair[1] then
		valErrorEpochPair[1] = valErr
		valErrorEpochPair[2] = epoch
	end
	logger:add{epoch .. "    " .. trainErr .. "    " ..  valErr}
end

print("Now testing on model no. " .. valErrorEpochPair[2] .. " with validation error= " .. valErrorEpochPair[1])
bestModelPath = paths.concat(opt.results,'model_'.. valErrorEpochPair[2] ..'.net')
evaluate( bestModelPath, testData, true)
