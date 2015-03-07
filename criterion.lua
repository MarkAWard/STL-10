-- define criterion / loss function to use

local C = {}

function C.select_criterion(options)

	if options.loss == 'margin' then

		-- This loss takes a vector of classes, and the index of
		-- the grountruth class as arguments. It is an SVM-like loss
		-- with a default margin of 1.

		criterion = nn.MultiMarginCriterion()

	elseif options.loss == 'nll' then

		-- This loss requires the outputs of the trainable model to
		-- be properly normalized log-probabilities, which can be
		-- achieved using a softmax function

		-- this should be be done in the model module
		-- model:add(nn.LogSoftMax())

		-- The loss works like the MultiMarginCriterion: it takes
		-- a vector of classes, and the index of the grountruth class
		-- as arguments.

		criterion = nn.ClassNLLCriterion()

	else
		error('unknown -model')
	end

	return criterion

end

return C
