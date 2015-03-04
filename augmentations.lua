
function translate(src, x, y)
    -- translate right x pixels  ~  torch.floor( torch.floor(2 * (#src)[2] * .1) * torch.rand(1)[1] - torch.floor((#src)[2] * .1))
    -- translate down y pixels   ~  torch.floor( torch.floor(2 * (#src)[3] * .1) * torch.rand(1)[1] - torch.floor((#src)[3] * .1))
    -- fill edges with the mirror of the image
    local dest = image.translate(src, x, y)
    -- mirror the left edge
    for j=1, (#src)[1] do
        for k=1, x do
            dest[j][{{}, {x - k + 1}}] = dest[j][{{}, {x + k}}]
        end
    end
    -- mirror the top edge
    for j=1, (#dest)[1] do
        for k=1, y do
            dest[j][{{y - k + 1}, {}}] = dest[j][{{y + k}, {}}]
        end
    end
    -- mirror the bottom edge
    for j=1, (#dest)[1] do
        for k=1, -y do
            dest[j][{{(#dest)[2] + y + k}, {}}] = dest[j][{{(#dest)[2] + y - k + 1}, {}}]
        end
    end
    -- mirror the right edge
    for j=1, (#dest)[1] do
        for k=1, -x do
            dest[j][{{}, {(#dest)[3] + x + k}}] = dest[j][{{}, {(#dest)[3] + x - k + 1}}]
        end
    end
    return dest
end

function scale(src, x, y, len)
    -- (x, y) bottom left corner
    -- (x+len, y+len) top right corner
    
    local width = (#src)[2]
    local height = (#src)[3]
    return image.scale(image.crop(src, x, y, x+len, y+len), width, height)
end

function rotation(src, deg) 
        
    local deg = deg * math.pi / 180.
	local expand = 100
    local new_i = torch.zeros((#src)[1], (#src)[2]+(2*expand), (#src)[3]+(2*expand))
    new_i[{{}, {expand+1, (#src)[2]+expand}, {expand+1, (#src)[3]+expand}}] = src
        
    for k=1, expand do
        new_i[{{}, {}, {expand - k + 1}}] = new_i[{{}, {}, {expand + k}}]
    end
    for k=1, expand do
        new_i[{{}, {(#new_i)[2] - expand + k}, {}}] = new_i[{{}, {(#new_i)[2] - expand - k + 1}, {}}]
    end
    for k=1, expand do
        new_i[{{}, {}, {(#new_i)[3] - expand + k}}] = new_i[{{}, {}, {(#new_i)[3] - expand - k + 1}}]
    end
    for k=1, expand do
        new_i[{{}, {expand - k + 1}, {}}] = new_i[{{}, {expand + k}, {}}]
    end
        
    new_i = image.rotate(new_i, deg)[{{}, {expand+1, (#src)[2]+expand}, {expand+1, (#src)[3]+expand}}]

    return new_i
    
end

function contrast2(src, p, m, c)
    -- p in [.25, 4]  ~  (4 - .25) * torch.rand(1)[1] + .25
    -- m in [.7, 1.4] ~  (1.4 - .7) * torch.rand(1)[1] + 1.4
    -- c in [-.1, .1] ~  (.1 + .1) * torch.rand(1)[1] - .1
    -- I think the c range is a little extreme, avoid negative numbers
    
    local dest = image.rgb2hsv(src)
    dest[1] = torch.pow(dest[1], p) * m + c
    dest[2] = torch.pow(dest[2], p) * m + c
    return image.hsv2rgb(dest)
end

function color_change(src, val) 
    -- val in [-.1, .1]  ~  val = (.1 + .1) * torch.rand(1)[1] - .1
    
    local dest = image.rgb2hsv(src)
    dest[1] = dest[1] + val
    return image.hsv2rgb(dest)
end

function augment()
	local translateParamX = torch.floor( torch.floor(2 * (#src)[2] * .1) * math.random(torch.rand(1)[1] - torch.floor((#src)[2] * .1))
	local translateParamY = torch.floor( torch.floor(2 * (#src)[3] * .1) * torch.rand(1)[1] - torch.floor((#src)[3] * .1))
	
	local scaleParam      = math.random(7, 14)/10. 
	
	scale
	rotation
	contrast2
	color_change