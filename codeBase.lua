require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
-- The type is by default 'double' so I leave it like this now as we never changed it before
-- When using CUDA
--  changes:  require 'cunn', the model, the criterion, input = input:cuda()
torch.setnumthreads( 8 )
torch.manualSeed(1) -- this was the default

---------------------------------------- READ DATA ------------------------------

--------------------------------------- END READ DATA ------------------------------

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

