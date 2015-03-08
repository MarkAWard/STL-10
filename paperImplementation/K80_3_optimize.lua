------------------------------- MAIN LEARNING FUNCTION ---------------------------------

-- creating symbolik links in the execution directory
os.execute('ln -s /tmp/elad/tr_bin.dat tr_bin.dat')
os.execute('ln -s /tmp/elad/ts_bin.dat ts_bin.dat')
os.execute('ln -s /tmp/elad/un_bin.dat un_bin.dat')

logger = optim.Logger(paths.concat('results', 'errorResults.log'))
logger:add{"EPOCH    TRAIN ERROR    VAL ERROR"}

valErrorEpochPair = {1.1,-1}
for epoch =1, opt.epochs do
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> EPOCH " .. epoch .. " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<") 
	trainErr = train( epoch )
	print(trainErr)
	valErr   = evaluate( paths.concat('results','model_'.. epoch ..'.net'), valData, false)
	if valErr < valErrorEpochPair[1] then
		valErrorEpochPair[1] = valErr
		valErrorEpochPair[2] = epoch
	end
	logger:add{epoch .. "    " .. trainErr .. "    " ..  valErr}
end
--[[print("Now testing on model no. " .. valErrorEpochPair[2] .. " with validation error= " .. valErrorEpochPair[1])
bestModelPath = paths.concat('results','model_'.. valErrorEpochPair[2] ..'.net')
evaluate( bestModelPath, testData, true) --]]