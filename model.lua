-- model definition module

local M = {}

function M.select_model(options)

	if options.model == 'cuda' then
		model = nn.Sequential()
		model:add(nn.SpatialConvolutionMM(3, 23, 7, 7, 2, 2, 2))
		model:add(nn.ReLU())
		model:add(nn.SpatialMaxPooling(3,3,2,2))
		model:add(nn.Dropout(.5))
		model:add(nn.Reshape(23*23*23))
		model:add(nn.Linear(23*23*23, 50))
		model:add(nn.Linear(50,10))
		model:add(nn.LogSoftMax())
	
	else
		model = nn.Sequential()
		model:add(nn.SpatialConvolution(3, 23, 7, 7, 2, 2))
		model:add(nn.ReLU())
		model:add(nn.SpatialMaxPooling(3,3,2,2))
		model:add(nn.Dropout(.5))
		model:add(nn.Reshape(23*22*22))
		model:add(nn.Linear(23*22*22, 50))
		model:add(nn.Linear(50,10))
		model:add(nn.LogSoftMax())
	end

	return model

end


return M