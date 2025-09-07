using DifferentialEquations
using Plots
using StaticArrays

const G = 6.67430e-11

struct TwoDParticle
    mass::Float64
    position::SVector{2,Float64}
    velocity::SVector{2,Float64}
end

function gravitational_force(p1::Particle, p2::Particle)
    r_vec = p2.position - p1.position
    r_mag = sqrt(sum(r_vec .^ 2))

    if r_mag < 1e-10
        return SVector(0.0, 0.0)
    end

    force_mag = G * p1.mass * p2.mass / (r_mag^2)

    return force_mag * r_vec
end

function two_body_dynamics!(du, u, p, t)
    pos1 = SVector(u[1], u[2])
    pos2 = SVector(u[3], u[4])
    vel1 = SVector(u[5], u[6])
    vel2 = SVector(u[7], u[8])

    m1, m2 = p

    particle1 = Particle(m1, pos1, vel1)
    particle2 = Particle(m2, pos2, vel2)

    force_on_1 = gravitational_force(particle1, particle2)
    force_on_2 = -force_on_1

    accel1 = force_on_1 / m1
    accel2 = force_on_2 / m2

    du[1:2] = vel1
    du[3:4] = vel2
    du[5:6] = accel1
    du[7:8] = accel2
end

function run_two_body_sim()
    # Initial conditions: []
    u0 = [
        0.0, 0.0,   # x1, y1
        1.0, 0.0,   # x2, y2
        0.0, 0.0,   # vx1, vy1
        0.0, 0.1    # vx2, vy2
    ]

    masses = (1e12, 1e12)

    tspan = (0.0, 200.0)

    prob = ODEProblem(two_body_dynamics!, u0, tspan, masses)
    sol = solve(prob, Tsit5(), saveat=0.1)

    return sol
end

# Visualization function
function plot_simulation(sol)
    # Extract trajectories
    x1 = [u[1] for u in sol.u]
    y1 = [u[2] for u in sol.u]
    x2 = [u[3] for u in sol.u]
    y2 = [u[4] for u in sol.u]

    # Create plot
    plot(
        title="Two-Body Gravitational Simulation",
        xlabel="X Position (m)",
        ylabel="Y Position (m)",
        aspect_ratio=:equal,
        legend=:topright
    )

    # Plot trajectories
    plot!(x1, y1, label="Particle 1", linewidth=2, color=:blue)
    plot!(x2, y2, label="Particle 2", linewidth=2, color=:red)

    # Mark starting positions
    scatter!([x1[1]], [y1[1]], label="Start 1", color=:blue, markersize=8)
    scatter!([x2[1]], [y2[1]], label="Start 2", color=:red, markersize=8)

    # Mark final positions
    scatter!([x1[end]], [y1[end]], label="End 1", color=:lightblue, markersize=8)
    scatter!([x2[end]], [y2[end]], label="End 2", color=:pink, markersize=8)
end

function create_animation_gif(sol, filename="orbital_motion.gif")
    println("Creating animation...")

    # Extract all trajectory data
    x1 = [u[1] for u in sol.u]
    y1 = [u[2] for u in sol.u]
    x2 = [u[3] for u in sol.u]
    y2 = [u[4] for u in sol.u]

    # Determine plot bounds
    all_x = vcat(x1, x2)
    all_y = vcat(y1, y2)
    margin = 0.1 * max(maximum(all_x) - minimum(all_x), maximum(all_y) - minimum(all_y))
    xlims = (minimum(all_x) - margin, maximum(all_x) + margin)
    ylims = (minimum(all_y) - margin, maximum(all_y) + margin)

    # Create animation
    anim = @animate for i in 1:length(sol.u)
        # Show trails (last 50 points)
        trail_start = max(1, i - 50)

        plot(
            title="Real-time Orbital Motion (t = $(round(sol.t[i], digits=1))s)",
            xlabel="X Position (m)", ylabel="Y Position (m)",
            xlims=xlims, ylims=ylims,
            aspect_ratio=:equal, size=(800, 600),
            background_color=:black, foreground_color=:white
        )

        # Plot trails
        if i > 1
            plot!(x1[trail_start:i], y1[trail_start:i],
                color=:cyan, alpha=0.6, linewidth=2, label="")
            plot!(x2[trail_start:i], y2[trail_start:i],
                color=:orange, alpha=0.6, linewidth=2, label="")
        end

        # Plot current particles
        scatter!([x1[i]], [y1[i]], color=:cyan, markersize=12,
            label="Massive Body", markerstrokewidth=2, markerstrokecolor=:white)
        scatter!([x2[i]], [y2[i]], color=:orange, markersize=8,
            label="Smaller Body", markerstrokewidth=2, markerstrokecolor=:white)

        # Add center of mass
        scatter!([0], [0], color=:yellow, markersize=4,
            label="Center of Mass", markershape=:star)
    end every 3  # Skip frames for faster creation

    # Save as GIF
    gif(anim, filename, fps=15)
    println("Animation saved as: $filename")

    return anim
end

## Run simulation
solution = run_two_body_sim()

## Create visualization
plot_simulation(solution)
create_animation_gif(solution)