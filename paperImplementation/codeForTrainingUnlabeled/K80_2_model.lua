model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3, 64, 7, 7, 1, 1, 2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2))
model:add(nn.Dropout(.5))
model:add(nn.SpatialConvolutionMM(64, 128, 5, 5, 1, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2))
model:add(nn.Dropout(.5))
model:add(nn.Reshape(128*22*22))
model:add(nn.Linear(128*22*22, 512))
model:add(nn.Linear(512, C))
model:add(nn.LogSoftMax())
--[[
	upper_part=nn.Sequential()
	upper_part:add(nn.SpatialConvolutionMM(3, 20, 7, 7, 2, 2, 2))
	upper_part:add(nn.ReLU())
	upper_part:add(nn.SpatialConvolutionMM(20, 30, 16, 16, 2, 2))
	upper_part:add(nn.ReLU())
	upper_part:add(nn.SpatialConvolutionMM(30, 30, 4, 4, 1, 1))
	upper_part:add(nn.ReLU())

	lower_part=nn.Sequential()
	lower_part:add(nn.SpatialConvolutionMM(3, 20, 7, 7, 6, 6, 2))
	lower_part:add(nn.ReLU())
	lower_part:add(nn.SpatialConvolutionMM(20, 30, 4, 4, 1, 1))
	lower_part:add(nn.ReLU())

	par=nn.Concat(2)
	par:add(upper_part)
	par:add(lower_part)

	model = nn.Sequential()
	model:add(par)
	model:add(nn.SpatialMaxPooling(4,4,2,2))
	model:add(nn.Dropout(.5))
	model:add(nn.Reshape(60*5*5))
	model:add(nn.Linear(60*5*5, 60))
	model:add(nn.Linear(60,10))
	model:add(nn.LogSoftMax())
	--]]
criterion = nn.ClassNLLCriterion()
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------- TRAIN FUNCTION --------------------------------------
function train( epoch )
	model:training() -- set model to training mode (for modules that differ in training and testing, like Dropout)
	-- Shuffling the training data   
	shuffle = torch.randperm(trainData:size())
	shuffed_tr_data=torch.zeros(trainData:size(),channels,sizeOfPatches, sizeOfPatches)
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
		--clr = 1/(1 + 3^epoch/math.exp(epoch) )
		parameters:add(-clr, gradParameters)
   end

   local filename = paths.concat('results', 'model_' .. epoch .. '.net')
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
	   local outputFile = paths.concat('results', 'output.csv')
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
    		for idx = 1, inputs:size()[1] do
    			f:write( t+idx-1 .. " , " .. argmax[idx][1] .. "\n") 
    		end
    	end 
    end
	if writeToFile then f:close() end
    return no_wrong/(dataset:size())
end
-------------------------------- END EVALUATE FUNCTION --------------------------------

