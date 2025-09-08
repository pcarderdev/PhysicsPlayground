# Physics Environment
mutable struct PhysicsWorld2D
    particles::Vector{Particle2D}

    bounds::Dict{String, Float64}   # "left", "right", "bottom", "top"
    
    gravity::Vector{Float64}        # [gx, gy]

    time::Float64
    dt::Float64
end

# PhysicsWorld2D constructor
function PhysicsWorld2D(width, height; gravity=[0.0, -9.81], dt=0.01)
    bounds = Dict("left" => 0.0, "right" => width, "bottom" => 0.0, "top" => height)

    return PhysicsWorld2D(Particle2D[], bounds, gravity, 0.0, dt)
end

# Add particles to PhysicsWorld2D
function add_particle!(world::PhysicsWorld2D, particle::Particle2D)
    push!(world.particles, particle)
end