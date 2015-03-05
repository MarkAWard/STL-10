------------------------------- MAIN LEARNING FUNCTION ---------------------------------
logger = optim.Logger(paths.concat('results', 'accuracyResults.log'))
logger:add{"EPOCH    TRAIN ERROR    VAL ERROR"}

valErrorEpochPair = {1.1,-1}
for i =1, opt.epochs do
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> EPOCH " .. i .. " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<") 
	trainErr = train(i)
	valErr   = evaluate( paths.concat('results','model_'..epoch..'.net'), valData, false)
	if valErr < valErrorEpochPair[1] then
		valErrorEpochPair[1] = valErr
		valErrorEpochPair[2] = i
	end
	logger:add{i .. "    " .. trainErr .. "    " ..  valErr}
end

bestModelPath = paths.concat('results','model_'.. valErrorEpochPair[2] ..'.net')
evaluate( bestModelPath, testData, true)