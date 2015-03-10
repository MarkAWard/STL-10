require 'torch'       -- torch
require 'image'       -- for color transforms
require 'nn'          -- provides a normalization operator
require 'optim'
local data = require 'data_preprocess'
local mod  = require 'model'
local crit = require 'criterion'
local aug  = require 'augmentations'
local optimize = require 'optimize'

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
cmd:option('-learningRate', 0.5, 'learning rate at t=0')
cmd:option('-lrDecay', 0.9, 'decrease learning rate at each epoch')
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
cmd:option('-warmStart', 'model_path', 'file path to a pre-trained model')
cmd:option('-mean', 'mean_path', 'file path to saved mean values for normalization')
cmd:option('-std', 'std_path', 'file path to saved std values for normalization')

cmd:text()
opt = cmd:parse(arg or {})

-- problem specific image size
opt.channels = 3
opt.imageHeight = 96
opt.imageWidth = 96

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
		allTrainData   = loadedTrain.X:t():reshape(opt.trainSize + opt.valSize, opt.channels, opt.imageHeight, opt.imageWidth)
		allTrainLabels = loadedTrain.y[1]
	end
	if opt.testSize ~= 0 then
		allTestData    = loadedTest.X:t():reshape(opt.testSize, opt.channels, opt.imageHeight, opt.imageWidth)
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
	trainData   = torch.zeros(number_of_images, opt.channels, opt.imageHeight, opt.imageWidth)
	trainLabels = torch.zeros(number_of_images)
end
if opt.valSize ~= 0 then
	valData     = torch.zeros(opt.valSize, opt.channels, opt.imageHeight, opt.imageWidth)
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
if opt.mean == 'mean_path' or opt.std == 'std_path' then
	mean, std = data.normalize_data(trainData, valData, testData)

	local filename = paths.concat(opt.results, 'mean.values')
	os.execute('mkdir -p ' .. sys.dirname(filename))
	torch.save(filename, mean)

	filename = paths.concat(opt.results, 'std.values')
	torch.save(filename, std)
else
	mean = torch.load(opt.mean)
	std  = torch.load(opt.std)
	if opt.trainSize ~= 0 then data.normalize(trainData, mean, std) end
	if opt.valSize ~= 0 then data.normalize(valData, mean, std) end
	if opt.testSize ~= 0 then data.normalize(testData, mean, std) end
end

print('==> setting model and criterion')
if opt.warmStart == 'model_path' then
	local model = mod.select_model(opt)
else
	print('    loading model ' .. opt.warmStart)
	local model = torch.load(opt.warmStart)
end
local criterion = crit.select_criterion(opt)

if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

print('Size of training data: ' .. trainData:size())
print(model)
print(criterion)

-- create logger file
logger = optim.Logger(paths.concat(opt.results, 'errorResults.log'))
logger:add{"EPOCH,TRAIN ERROR,VAL ERROR"}

valErrorEpochPair = {1.1,-1}
for epoch = 1, opt.epochs do

	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> EPOCH " .. epoch .. " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<") 

	-- train model
	trainErr = optimize.train( model, criterion, trainData, opt, epoch )
	print(trainErr)

	-- calculate validation error
	valErr = optimize.evaluate( paths.concat(opt.results,'model_'.. epoch ..'.net'), valData, false, opt)
	if valErr < valErrorEpochPair[1] then
		valErrorEpochPair[1] = valErr
		valErrorEpochPair[2] = epoch
	end
	-- write train and validation errors
	logger:add{epoch .. "," .. trainErr .. "," ..  valErr}
end

-- go back to the model with the best validation error and create predictions on test set
print("Now testing on model no. " .. valErrorEpochPair[2] .. " with validation error= " .. valErrorEpochPair[1])
bestModelPath = paths.concat(opt.results,'model_'.. valErrorEpochPair[2] ..'.net')
evaluate( bestModelPath, testData, true, opt)




