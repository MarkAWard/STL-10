if opt.type == 'cuda' then
   model = nn.Sequential()
   model:add(nn.SpatialConvolutionMM(3, 23, 7, 7, 2, 2, 2))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(3,3,2,2))
   model:add(nn.Dropout(.5))
   model:add(nn.Reshape(23*22*22))
   model:add(nn.Linear(23*22*22, 50))
   model:add(nn.Linear(50,10))
   model:add(nn.LogSoftMax())
else
   -- the model is not the updated one we use when the CUDA flag is on
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

criterion = nn.ClassNLLCriterion()

if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end


----------------------------------- TRAIN FUNCTION --------------------------------------

function train( epoch )
	classes = {'1','2','3','4','5','6','7','8','9','0'}
	local confusion = optim.ConfusionMatrix(classes)

   model:training()    -- set model to training mode (for modules that differ in training and testing, like Dropout)
   --randperm does not support cuda
   
   
   -- BEGIN: DO RESHUFFLE
   
   if opt.type=='cuda' then
   torch.setdefaulttensortype('torch.FloatTensor')
   shuffle = torch.randperm(trainData:size()) -- shuffle at each epoch
   shuffle=shuffle:cuda()
   torch.setdefaulttensortype('torch.CudaTensor')
   -- end of manuaver
   else shuffle = torch.randperm(trainData:size()) end
   
  shuffed_tr_data=torch.zeros(trainData:size(),channels,imageHeight,imageWidth)
  shuffed_tr_targets=torch.zeros(trainData:size())

   for t = 1, trainData:size() do
   
      shuffed_tr_data[t]=trainData.data[shuffle[t]]
      shuffed_tr_targets[t]=trainData.labels[shuffle[t]]
	  end
    -- END: DO RESHUFFLE

--BEGIN code from Chris for BATCH TRAINING 
	 
	parameters,gradParameters = model:getParameters()
   local clr = 0.1
   local no_wrong=0
   for t = 1,trainData:size(),opt.batchSize do
	  local inputs  = shuffed_tr_data[{{t, math.min(t+opt.batchSize-1, trainData:size())}}]
	  local targets = shuffed_tr_targets[{{t, math.min(t+opt.batchSize-1, trainData:size())}}]
	  
	  if opt.type=='cuda' then 
	  inputs=inputs:cuda()
	  targets=targets:cuda()
	  end
	  
	  
	  gradParameters:zero()
	  
	  --print(torch.type(inputs))
	  --print(#inputs)
	  
	  local output = model:forward(inputs)
	  
	  --print(torch.type(targets))
	  --print(#targets)	  
	  --print(torch.type(output))
	  --print(#output)
	  
	  local f = criterion:forward(output, targets)
	  local trash, argmax = output:max(2)
	  
	  if opt.type=='cuda' then 
	  argmax=argmax:cuda() else 
	  argmax=argmax:float() end
	  
	  
      no_wrong = no_wrong + torch.ne(argmax, targets):sum()
      model:backward(inputs, criterion:backward(output, targets))
	  
--      clr = opt.learningRate * (0.5 ^ math.floor(epoch / opt.lrDecay))
      clr = opt.learningRate
	  
      parameters:add(-clr, gradParameters)
	  
	  
	  
	  argmax=argmax:reshape((#inputs)[1])
	  
	  confusion:batchAdd(argmax, targets)

   end

   local filename = paths.concat('results', 'model_' .. epoch .. '.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   torch.save(filename, model)
   print(confusion)
   p=no_wrong/(trainData:size())
   print(p)
   return confusion.totalValid*100
end
--------------------------------- END TRAIN FUNCTION --------------------------------

-------------------------------- EVALUATE FUNCTION --------------------------------------
function evaluate( dataset )
   classes = {'1','2','3','4','5','6','7','8','9','0'}
   local confusion_val = optim.ConfusionMatrix(classes)
   model:evaluate()
   for t = 1,dataset:size() do
      local input = dataset.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = dataset.labels[t]
      local pred = model:forward(input)
      confusion_val:add(pred, target)
   end
   return confusion_val.totalValid * 100
end
--------------------------------- END VAL FUNCTION --------------------------------
