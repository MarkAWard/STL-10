-- model definition module

local M = {}

function M.select_model(options)

	if options.model == 'par' then

		upper_part=nn.Sequential()
		upper_part:add(nn.SpatialConvolutionMM(3, 20, 7, 7, 2, 2, 2))
		upper_part:add(nn.ReLU())
		upper_part:add(nn.SpatialMaxPooling(2,2,2,2))

		upper_part:add(nn.SpatialConvolutionMM(20, 30, 4, 4, 2, 2))
		upper_part:add(nn.ReLU())

		
		lower_part=nn.Sequential()
		lower_part:add(nn.SpatialConvolutionMM(3, 20, 27, 27, 2, 2, 2))
		lower_part:add(nn.ReLU())
		lower_part:add(nn.SpatialMaxPooling(3,3,2,2))
		
		lower_part:add(nn.SpatialConvolutionMM(20, 30, 9, 9, 1, 1))
		lower_part:add(nn.ReLU())
		
		
		par=nn.Concat(2)
		par:add(upper_part)
		par:add(lower_part)

		model = nn.Sequential()
		model:add(par)
		model:add(nn.SpatialMaxPooling(3,3,2,2))
		model:add(nn.Dropout(.5))
		model:add(nn.Reshape(60*4*4))
		model:add(nn.Linear(60*4*4, 100))
		--model:add(nn.ReLU())
		model:add(nn.Linear(100,10))
		model:add(nn.LogSoftMax())

	elseif opt.model == 'deep' then

		model = nn.Sequential()

		model:add(nn.SpatialConvolutionMM(3, 64, 7, 7, 1, 1, 2))
		model:add(nn.ReLU())
		model:add(nn.SpatialMaxPooling(2,2))
		model:add(nn.Dropout(.5))

		model:add(nn.SpatialConvolutionMM(64, 128, 5, 5, 1, 1, 1))
		model:add(nn.ReLU())
		model:add(nn.SpatialMaxPooling(2,2))
		model:add(nn.Dropout(.5))

		model:add(nn.SpatialConvolutionMM(128, 256, 5, 5, 1, 1))
		model:add(nn.ReLU())
		model:add(nn.SpatialMaxPooling(2,2))
		model:add(nn.Dropout(.5))

		model:add(nn.Reshape(256 * 9 * 9))
		model:add(nn.Linear(256 * 9 * 9, 512))
		model:add(nn.Linear(512, 10))

		model:add(nn.LogSoftMax())

	else
	   model = nn.Sequential()
	   model:add(nn.SpatialConvolutionMM(3, 23, 7, 7, 2, 2))
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
