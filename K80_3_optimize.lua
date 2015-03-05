------------------------------- MAIN LEARNING FUNCTION ---------------------------------
logger = optim.Logger(paths.concat('results', 'accuracyResults.log'))
logger:add{"EPOCH    TRAIN ERROR    VAL ERROR"}

valErrorEpochPair = {1.1,-1}
for epoch =1, opt.epochs do
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> EPOCH " .. epoch .. " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<") 
	trainErr = train( epoch )
	valErr   = evaluate( paths.concat('results','model_'.. epoch ..'.net'), valData, false)
	if valErr < valErrorEpochPair[1] then
		valErrorEpochPair[1] = valErr
		valErrorEpochPair[2] = epoch
	end
	logger:add{epoch .. "    " .. trainErr .. "    " ..  valErr}
end

print("Now testing on model no. " .. valErrorEpochPair[2] .. " with validation error= " .. valErrorEpochPair[1])
bestModelPath = paths.concat('results','model_'.. valErrorEpochPair[2] ..'.net')
evaluate( bestModelPath, testData, true)