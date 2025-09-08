# Core particle with physics properties
mutable struct Particle2D
    
    # Kinematics
    position::Vector{Float64}
    velocity::Vector{Float64}
    acceleration::Vector{Float64}

    # Properties
    mass::Float64

end

# Particle2D constructor
function Particle2D(x, y, vx, vy; mass=1.0)
    pos = [x, y]
    vel = [vx, vy]
    acc = [0.0, 0.0]

    return Particle2D(pos, vel, acc, mass)
end