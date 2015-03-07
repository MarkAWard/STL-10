require 'torch'       -- torch
require 'image'       -- for color transforms
require 'nn'          -- provides a normalization operator
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
cmd:option('-type', 'double', 'type: double | float | cuda')

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

cmd:text()
opt = cmd:parse(arg or {})

-- problem specific image size
channels = 3
imageHeight = 96
imageWidth = 96

-- set environment and defaults
torch.setnumthreads( opt.threads )
torch.manualSeed( opt.seed )
if opt.type == 'float' then
	print('==> switching to floats')
	torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
	print('==> switching to CUDA')
	torch.setdefaulttensortype('torch.FloatTensor')
	require 'cunn'
	-- IS THIS ONLY FOR K80????
	if opt.machine == 'k80' then
		cutorch.setDevice(3)
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
	trainData   = torch.zeros(opt.trainSize, channels, imageHeight, imageWidth)
	trainLabels = torch.zeros(opt.trainSize)
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

-- create final data objects
-- $$$$$$$$$ TODO add in extra data $$$$$$$$$$$$$
if opt.trainSize ~= 0 then
	trainData = {
	   data   = trainData,
	   labels = trainLabels,
	   size = function() return opt.trainSize end
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

print(model)
print(criterion)


