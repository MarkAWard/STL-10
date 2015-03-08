-- training and testing functions

local F = {}

function F.train(model, criterion, trainData, options, epoch)

	-- set model to training mode (for modules that differ in training and testing, like Dropout)
	model:training() 

	-- Shuffling the training data   
	shuffle            = torch.randperm(trainData:size())
	shuffed_tr_data    = torch.zeros(trainData:size(), options.channels, options.imageHeight, options.imageWidth)
	shuffed_tr_targets = torch.zeros(trainData:size())	
	for t = 1, trainData:size() do
		shuffed_tr_data[t] = trainData.data[shuffle[t]]
		shuffed_tr_targets[t] = trainData.labels[shuffle[t]]
	end
	
	-- batch training to exploit CUDA optimizations
	parameters, gradParameters = model:getParameters()
	local no_wrong=0
	for t = 1, trainData:size(), options.batchSize do
		-- create the batch
		local inputs  = shuffed_tr_data[ {{ t, math.min(t+options.batchSize-1, trainData:size()) }} ]
		local targets = shuffed_tr_targets[ {{ t, math.min(t+options.batchSize-1, trainData:size()) }} ]
		if options.type == 'cuda' then 
			inputs  = inputs:cuda()
			targets = targets:cuda()
		end
		gradParameters:zero()
		
		-- forward and backward passes
		local output = model:forward(inputs)
		local f = criterion:forward(output, targets)
		local trash, argmax = output:max(2)
	  	if options.type =='cuda' then  
	  		argmax = argmax:cuda() 
	  	else 
	  		argmax = argmax:float() 
	  	end
	  	
	  	no_wrong = no_wrong + torch.ne(argmax, targets):sum()
	  	model:backward(inputs, criterion:backward(output, targets))

		-- clr = options.learningRate * (0.5 ^ math.floor(epoch / options.lrDecay))
		clr = options.learningRate * (options.lrDecay ^ (epoch - 1) )
				
		parameters:add(-clr, gradParameters)
		
   end

   -- save the model
   local filename = paths.concat(options.results, 'model_' .. epoch .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   torch.save(filename, model)

   return no_wrong/(trainData:size())   

end


function F.evaluate( modelPath, dataset, writeToFile, options)
	-- load the model
	modelToEval = torch.load(modelPath)

	-- save predictions to file
	local f
	if writeToFile then
	   local outputFile = paths.concat(options.results, 'output.csv')
	   f = io.open(outputFile, "w")
	   f:write("Id,Category\n")
	end
	
	modelToEval:evaluate()
	local no_wrong = 0
	for t = 1, dataset:size(), options.batchSize do
		-- create batches
		local inputs  = dataset.data[ {{ t, math.min(t+options.batchSize-1, dataset:size()) }} ]
		local targets = dataset.labels[ {{ t, math.min(t+options.batchSize-1, dataset:size()) }} ]
		if options.type == 'cuda' then 
			inputs  = inputs:cuda() 
			targets = targets:cuda()
    	end
    	local output = modelToEval:forward(inputs)
    	local trash, argmax = output:max(2)
    	no_wrong = no_wrong + torch.ne(argmax, targets):sum()
    	
    	-- write predictions
    	if writeToFile then
    		for idx = 1, options.batchSize do
    			f:write( t+idx-1 .. " , " .. argmax[idx][1] .. "\n") 
    		end
    	end 

    end
	if writeToFile then f:close() end

	-- return error
    return no_wrong/(dataset:size())

end


return F
