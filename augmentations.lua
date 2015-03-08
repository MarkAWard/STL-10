-- data augmentation module
require 'image'

local A = {} --public interface

function A.translate(src, x, y)
    -- translate right x pixels  ~  math.random(-math.ceil((#src)[2] * .15), math.ceil((#src)[2] * .15))
    -- translate down y pixels   ~  math.random(-math.ceil((#src)[3] * .15), math.ceil((#src)[3] * .15))
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

function A.scale(src, x, y, len)
    -- (x, y) in [1, ceil((#src)[2] * .25)], [1, ceil((#src)[3] * .25)]
    -- (x+len, y+len) 
    
    local width = (#src)[2]
    local height = (#src)[3]
    return image.scale(image.crop(src, x, y, x+len, y+len), width, height)
end

function A.rotation(src, deg) 
    -- deg in [-20, 20] ~ (20. + 20.) * torch.rand(1)[1] - 20.

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

function A.contrast(src, p, m, c)
    -- p in [.25, 4]  ~  (4 - .25) * torch.rand(1)[1] + .25
    -- m in [.7, 1.4] ~  (1.4 - .7) * torch.rand(1)[1] + 1.4
    -- c in [-.1, .1] ~  (.1 + .1) * torch.rand(1)[1] - .1
    -- I think the c range is a little extreme, avoid negative numbers
    
    local dest = image.rgb2hsv(src)
    dest[1] = torch.pow(dest[1], p) * m + c
    dest[2] = torch.pow(dest[2], p) * m + c
    return image.hsv2rgb(dest)
end

function A.color_change(src, val) 
    -- val in [-.1, .1]  ~  val = (.1 + .1) * torch.rand(1)[1] - .1
    
    local dest = image.rgb2hsv(src)
    dest[1] = dest[1] + val
    return image.hsv2rgb(dest)
end

function A.augment(src, options)

    local dest = src

    if torch.rand(1)[1] < options.flip then
        dest = image.hflip(dest)
    end

    -- translate
    if torch.rand(1)[1] < options.translate then
        local x = math.random(-math.ceil((#dest)[2] * .15), math.ceil((#dest)[2] * .15))
        local y = math.random(-math.ceil((#dest)[3] * .15), math.ceil((#dest)[3] * .15))
        dest = A.translate(dest, x, y)
    end

    -- scale
    if torch.rand(1)[1] < options.scale then
        local x = math.random(1, math.ceil((#dest)[2] * .15))
        local y = math.random(1, math.ceil((#dest)[3] * .15))
        local r = math.max(x, y)
        local len = math.random(math.floor(.8 * (#dest)[2]) - math.floor(.8 * r), (#dest)[2] - r)
        dest = A.scale(dest, x, y, len)
    end

    -- rotate
    if torch.rand(1)[1] < options.rotate then
        local deg = (20. + 20.) * torch.rand(1)[1] - 20.
        dest = A.rotation(dest, deg)
    end

    -- contrast
    if torch.rand(1)[1] < options.contrast then
        local p = (4 - .25) * torch.rand(1)[1] + .25
        local m = (1.4 - .7) * torch.rand(1)[1] + 1.4
        local c = (.1 + .1) * torch.rand(1)[1] - .1
        dest = A.contrast(dest, p, m, c)
    end

    -- color
    if torch.rand(1)[1] < options.color then
        local val = (.1 + .1) * torch.rand(1)[1] - .1
        dest = A.color_change(dest, val)
    end

    return dest
end

return A




