--[[
That's the model where we trained the unlabeled data
model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3, 23, 7, 7, 2, 2, 2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3,3,2,2))
model:add(nn.Dropout(.5))
model:add(nn.Reshape(23*7*7))
model:add(nn.Linear(23*7*7, 50))
model:add(nn.Linear(50, C))
model:add(nn.LogSoftMax()) 
--]]  
model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(23, 40, 3, 3, 1, 1))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,1,1))
model:add(nn.Reshape(40*4*4))
model:add(nn.Linear(40*4*4, 60))
model:add(nn.Linear(60,10))
model:add(nn.LogSoftMax()) 



criterion = nn.ClassNLLCriterion()
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------- TRAIN FUNCTION --------------------------------------

function trainWithUnlabeledModel( epoch, unlaModelPath )
	unlaModel = torch.load(unlaModelPath)
	unlaModel:evaluate()
	
	newInput = torch.zeros( trainData:size(), 23,7,7 )
	trainIdx = 1
	for t = 1,trainData:size(), opt.batchSize do
		local inputs  = trainData.data[{{t, math.min(t+opt.batchSize-1, trainData:size())}}]
		local sizeBatchSample = inputs:size()[1]
		if opt.type == 'cuda' then 
			inputs  = inputs:cuda() 
    	end
		unlaModel:forward(inputs)
    	for idx = 1, sizeBatchSample do
	    	newInput[trainIdx] = unlaModel:get(5).output[idx]:float()
	    	trainIdx = trainIdx+1
    	end
    end

	model:training() -- set model to training mode (for modules that differ in training and testing, like Dropout)
	-- Shuffling the training data   
	shuffle = torch.randperm(trainData:size())
	shuffed_tr_data=torch.zeros(trainData:size(), 23, 7, 7)
	shuffed_tr_targets=torch.zeros(trainData:size())	
	for t = 1, trainData:size() do
		shuffed_tr_data[t]=newInput[shuffle[t]]
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
function evaluateUnlabeledVersion( modelPath, dataset, writeToFile, unlaModelPath)
	local f
	if writeToFile then
	   local outputFile = paths.concat('results', 'output.csv')
	   f = io.open(outputFile, "w")
	   f:write("Id,Category\n")
	end
	
	local modelToEval = torch.load(modelPath)
	local unlModel = torch.load(unlaModelPath)
	unlModel:evaluate()
	
	newinp = torch.zeros( dataset:size(), 23,7,7 )
	evalIdx = 1
	for t = 1,dataset:size(), opt.batchSize do
		local inputs  = dataset.data[{{t, math.min(t+opt.batchSize-1, dataset:size())}}]
		local sizeBatchSample = inputs:size()[1]
		if opt.type == 'cuda' then 
			inputs  = inputs:cuda() 
    	end
		unlModel:forward(inputs)
    	for idx = 1, sizeBatchSample do
	    	newinp[evalIdx] = unlModel:get(5).output[idx]:float()
	    	evalIdx = evalIdx+1
    	end
    end
	
	modelToEval:evaluate()
	local no_wrong = 0
	for t = 1,dataset:size(), opt.batchSize do
		local inputs  = newinp[{{t, math.min(t+opt.batchSize-1, dataset:size())}}]
		local sizeBatchSample = inputs:size()[1]
		local targets = dataset.labels[{{t, math.min(t+opt.batchSize-1, dataset:size())}}]
		if opt.type == 'cuda' then 
			inputs  = inputs:cuda() 
			targets = targets:cuda()
    	end
    	local output = modelToEval:forward(inputs)
    	local trash, argmax = output:max(2)
    	no_wrong = no_wrong + torch.ne(argmax, targets):sum()
    	
    	if writeToFile then
    		for idx = 1, sizeBatchSample do
    			f:write( t+idx-1 .. " , " .. argmax[idx][1] .. "\n") 
    		end
    	end 
    	 	
    end
	if writeToFile then f:close() end
    return no_wrong/(dataset:size())
end

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

