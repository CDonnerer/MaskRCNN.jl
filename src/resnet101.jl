using Flux
using Flux: @treelike

struct ConvNorm
  conv
  norm
  activation
end

@treelike ConvNorm

function ConvNorm(size, channels; activation = relu)
    ConvNorm(
        Conv(size, channels),
        BatchNorm(channels[2]),
        x -> activation.(x)
    )
end

# Overload call, so the object can be used as a function
(m::ConvNorm)(x) = m.activation(m.norm(m.conv(x)))

model = Chain(
        ConvNorm((1,1), 3=>6, activation=relu),
        ConvNorm((2,2), 6=>6, activation=relu),
        ConvNorm((1,1), 6=>6, activation=identity),
)

img = reshape(
    [1 1 1 1 0 0 0 0 0 0 0 0], 2, 2, 3, 1
)

conv_img = model(img)


function reslayerx0(w,x,ms; padding=0, stride=1, mode=1)
    b  = conv4(w[1],x; padding=padding, stride=stride)
    bx = batchnorm(w[2:3],b,ms; mode=mode)
end

function reslayerx1(w,x,ms; padding=0, stride=1, mode=1)
    relu.(reslayerx0(w,x,ms; padding=padding, stride=stride, mode=mode))
end

function reslayerx2(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    ba = reslayerx1(w[1:3],x,ms; padding=pads[1], stride=strides[1], mode=mode)
    bb = reslayerx1(w[4:6],ba,ms; padding=pads[2], stride=strides[2], mode=mode)
    bc = reslayerx0(w[7:9],bb,ms; padding=pads[3], stride=strides[3], mode=mode)
end


function reslayerx4(w,x,ms; pads=[0,1,0], strides=[1,1,1], mode=1)
    relu.(x .+ reslayerx2(w,x,ms; pads=pads, strides=strides, mode=mode))
end
