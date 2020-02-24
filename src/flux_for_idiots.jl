

function linear(in, out)
    W = param(randn(out, in))
    b = param(randn(out))
    x -> W * x .+ b
end

linear1 = linear(5, 3)
