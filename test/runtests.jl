import MaskRCNN
using Test


@testset "MaskRCNN.jl" begin
    @test MaskRCNN.my_fun_func(2, 1) == 5
end
