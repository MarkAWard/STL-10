-- data loading and preprocessing module
require 'image'

local D = {}

function D.normalize_data(train, val, test) 

	print('    converting to yuv...')
	train.data = train.data:float()
	for i = 1, train:size() do
		train.data[i] = image.rgb2yuv(train.data[i])
	end

	if val then
		val.data = val.data:float()
		for i = 1,val:size() do
			val.data[i]	= image.rgb2yuv(val.data[i])
		end
	end

	if test then
		test.data = test.data:float()
		for i = 1, test:size() do
			test.data[i] = image.rgb2yuv(test.data[i])
		end
	end

	channelsYUV = {'y','u','v'}
	mean = {}
	std = {}

	print('    normalizing globally...')
	-- normalize each channel globally
	for i,channel in ipairs(channelsYUV) do
		mean[i] = train.data[{ {},i,{},{} }]:mean()
		std[i] = train.data[{ {},i,{},{} }]:std()
		train.data[{ {},i,{},{} }]:add(-mean[i])
		train.data[{ {},i,{},{} }]:div(std[i])
	end

	if val then
		for i,channel in ipairs(channelsYUV) do
			-- Normalize val, test data, using the training means/stds
			val.data[{ {},i,{},{} }]:add(-mean[i])
			val.data[{ {},i,{},{} }]:div(std[i])
			if test then 
				test.data[{ {},i,{},{} }]:add(-mean[i])
				test.data[{ {},i,{},{} }]:div(std[i])
			end
		end
	end
	
	print('    normalizing locally...')
	-- Normalize all three channels locally
	neighborhood = image.gaussian1D(13)
	normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
	for c in ipairs(channelsYUV) do
		
		for i = 1,train:size() do
			train.data[{ i,{c},{},{} }] = normalization:forward(train.data[{ i,{c},{},{} }])
		end

		if val then
			for i = 1,val:size() do
				val.data[{ i,{ c},{},{} }] = normalization:forward(val.data[{ i,{c},{},{} }])
			end
		end

		if test	then
			for i = 1,test:size() do
				test.data[{ i,{c},{},{} }] = normalization:forward(test.data[{ i,{c},{},{} }])
			end
		end

	end

	return mean, std

end


function D.normalize(dataset, mean, std)

	dataset.data = dataset.data:float()
	for i = 1, dataset:size() do
		dataset.data[i] = image.rgb2yuv(dataset.data[i])
	end

	-- normalize each channel globally
	channelsYUV = {'y','u','v'}
	for i,channel in ipairs(channelsYUV) do
		dataset.data[{ {},i,{},{} }]:add(-mean[i])
		dataset.data[{ {},i,{},{} }]:div(std[i])
	end

	-- Normalize all three channels locally
	neighborhood = image.gaussian1D(13)
	normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

	for c in ipairs(channelsYUV) do	
		for i = 1,dataset:size() do
			dataset.data[{ i,{c},{},{} }] = normalization:forward(dataset.data[{ i,{c},{},{} }])
		end
	end

end


return D





