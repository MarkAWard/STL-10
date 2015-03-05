------------------------------- MAIN LEARNING FUNCTION ---------------------------------
logger = optim.Logger(paths.concat('results', 'accuracyResults.log'))
logger:add{"EPOCH  TRAIN ACC  VAL ACC"}


for i =1, opt.epochs do
	print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>> EPOCH " .. i .. " <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<") 
	trainAcc = train(i)
	valAcc   = evaluate(valData)
	logger:add{i .. "," .. trainAcc .. "," ..  valAcc}
end
