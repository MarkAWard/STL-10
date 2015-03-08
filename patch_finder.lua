
-- x is an image
-- w is the window of the patch
function patch_finder(x,w)

	model=nn.SpatialAveragePooling(w,w,1,1)
	--x=image.rotate(trainData.data[k],-1.5707963268)
	x_grad=image.rgb2y(image.convolve(x, image.laplacian(8)))
	x_grad=torch.abs(x_grad)
	 
	output=model:forward(x_grad)
	max_val=torch.max(output)

    
    --print(#output)

	for i=1, (#output)[2] do
	    for j=1, (#output)[2] do
		holder=output[{{1},{i},{j}}]:reshape(1)
		if holder[1]==max_val then
		    tmp={i,j} end
		end
	end

    --print(tmp)
	return(x[{{},{tmp[1],tmp[1]+w},{tmp[2],tmp[2]+w}}])
end
	












