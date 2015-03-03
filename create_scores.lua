require 'torch'
require 'image'
require 'nn'  
require 'mattorch'
require 'optim'

torch.setnumthreads( 8 )

trainSize     = 4500
valSize       = 500
testSize      = 8000
extraSize     = 100000
channels      = 3
imageHeight   = 96
imageWidth    = 96
outputClasses = 10

trainFile = '/scratch/courses/DSGA1008/A2/matlab/train.mat'
testFile = '/scratch/courses/DSGA1008/A2/matlab/test.mat'

loadedTrain = mattorch.load(trainFile)
loadedTest = mattorch.load(testFile)
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

function val()
   classes = {'1','2','3','4','5','6','7','8','9','0'}
   confusion = optim.ConfusionMatrix(classes)
   model:evaluate()
   for t = 1,valData:size() do
      local input = valData.data[t]
      input = input:double()
      local target = valData.labels[t]
      local pred = model:forward(input)
      confusion:add(pred, target)
   end
   print(confusion)
   return confusion.totalValid * 100
end

function train()
   classes = {'1','2','3','4','5','6','7','8','9','0'}
   confusion = optim.ConfusionMatrix(classes)
   model:evaluate()
   for t = 1,trainData:size() do
      local input = trainData.data[t]
      input = input:double()
      local target = trainData.labels[t]
      local pred = model:forward(input)
      confusion:add(pred, target)
   end
   print(confusion)
   return confusion.totalValid * 100
end

f = io.open("accuracy.csv", "w")
f:write("i,train,val\n")
for i = 1, 125 do
    model = torch.load("/scratch/maw627/logs/A2-3252370/results/model_" .. i .. ".net")
    model:evaluate()
    print(">>>>>>>>>>>>>>>>>>>> MODEL " .. i .. " <<<<<<<<<<<<<<<<<<<<<<")
    trainAcc = train()
    valAcc = val()
    f:write(i .. "," .. trainAcc .. "," .. valAcc .. "\n")
end
f:close()