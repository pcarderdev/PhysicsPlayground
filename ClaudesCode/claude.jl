using DifferentialEquations
using Plots
using StaticArrays

# Physics constants
const G = 6.67430e-11  # Gravitational constant (m³/kg·s²)

# CPU-optimized particle structure using 3D StaticArrays
struct Particle3D
    mass::Float64
    position::SVector{3,Float64}  # Changed to 3D
    velocity::SVector{3,Float64}  # Changed to 3D
end

# CPU-optimized gravitational force calculation for 3D
@inline function gravitational_force_3d(p1::Particle3D, p2::Particle3D)
    r_vec = p2.position - p1.position
    r_mag_sq = sum(abs2, r_vec)  # Works for 3D

    # Avoid division by zero with softening
    if r_mag_sq < 1e-20
        return SVector(0.0, 0.0, 0.0)  # 3D zero vector
    end

    r_mag = sqrt(r_mag_sq)
    # Force magnitude: F = G * m1 * m2 / r²
    force_mag = G * p1.mass * p2.mass / (r_mag_sq * r_mag)

    # Force vector (on p1 due to p2)
    return force_mag * r_vec
end

# System dynamics function optimized for 3D CPU performance
function two_body_dynamics_3d!(du, u, p, t)
    # Extract positions and velocities for 3D
    # u = [x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2]

    pos1 = SVector(u[1], u[2], u[3])      # 3D position
    pos2 = SVector(u[4], u[5], u[6])      # 3D position
    vel1 = SVector(u[7], u[8], u[9])      # 3D velocity
    vel2 = SVector(u[10], u[11], u[12])   # 3D velocity

    # Get masses from parameters
    m1, m2 = p

    # Create particle objects (stack allocated)
    particle1 = Particle3D(m1, pos1, vel1)
    particle2 = Particle3D(m2, pos2, vel2)

    # Calculate forces
    force_on_1 = gravitational_force_3d(particle1, particle2)
    force_on_2 = -force_on_1  # Newton's third law

    # Calculate accelerations (F = ma → a = F/m)
    accel1 = force_on_1 / m1
    accel2 = force_on_2 / m2

    # Set derivatives (in-place for performance) - 3D version
    du[1] = vel1[1]      # dx1/dt = vx1
    du[2] = vel1[2]      # dy1/dt = vy1
    du[3] = vel1[3]      # dz1/dt = vz1
    du[4] = vel2[1]      # dx2/dt = vx2
    du[5] = vel2[2]      # dy2/dt = vy2
    du[6] = vel2[3]      # dz2/dt = vz2
    du[7] = accel1[1]    # dvx1/dt = ax1
    du[8] = accel1[2]    # dvy1/dt = ay1
    du[9] = accel1[3]    # dvz1/dt = az1
    du[10] = accel2[1]   # dvx2/dt = ax2
    du[11] = accel2[2]   # dvy2/dt = ay2
    du[12] = accel2[3]   # dvz2/dt = az2
end

# Setup initial conditions for 3D orbital motion
function run_two_body_simulation_3d()
    # Calculate circular orbit parameters
    total_mass = 2e12  # kg
    m1, m2 = total_mass * 0.6, total_mass * 0.4  # Unequal masses
    separation = 2.0   # meters

    # Center of mass system
    r1 = separation * m2 / (m1 + m2)
    r2 = separation * m1 / (m1 + m2)

    # Circular orbital velocity
    v_orbit = sqrt(G * (m1 + m2) / separation)

    # 3D Initial conditions: orbital motion in XY plane
    # [x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2]
    u0 = [
        -r1, 0.0, 0.0,                           # Particle 1 position (3D)
        r2, 0.0, 0.0,                           # Particle 2 position (3D)
        0.0, -v_orbit * m2 / (m1 + m2), 0.0,       # Particle 1 velocity (3D)
        0.0, v_orbit * m1 / (m1 + m2), 0.0        # Particle 2 velocity (3D)
    ]

    # Time span (seconds)
    tspan = (0.0, 200.0)

    # Create and solve ODE problem with high-performance solver
    prob = ODEProblem(two_body_dynamics_3d!, u0, tspan, (m1, m2))

    # Use Vern7 for high accuracy with good performance
    sol = solve(prob, Vern7(), abstol=1e-12, reltol=1e-10, saveat=0.1)

    return sol
end

# Enhanced 3D visualization function
function plot_simulation_3d(sol)
    # Extract 3D trajectories
    x1 = [u[1] for u in sol.u]
    y1 = [u[2] for u in sol.u]
    z1 = [u[3] for u in sol.u]
    x2 = [u[4] for u in sol.u]
    y2 = [u[5] for u in sol.u]
    z2 = [u[6] for u in sol.u]

    # Create 3D plot
    p = plot(
        title="3D Two-Body Gravitational System",
        xlabel="X Position (m)",
        ylabel="Y Position (m)",
        zlabel="Z Position (m)",
        size=(800, 600),
        dpi=150
    )

    # Plot 3D trajectories
    plot!(p, x1, y1, z1, label="Massive Body", linewidth=3, color=:blue, alpha=0.8)
    plot!(p, x2, y2, z2, label="Smaller Body", linewidth=2, color=:red, alpha=0.8)

    # Mark starting positions
    scatter!(p, [x1[1]], [y1[1]], [z1[1]], label="Start 1", color=:darkblue, markersize=10)
    scatter!(p, [x2[1]], [y2[1]], [z2[1]], label="Start 2", color=:darkred, markersize=8)

    # Mark final positions
    scatter!(p, [x1[end]], [y1[end]], [z1[end]], label="End 1", color=:lightblue, markersize=10)
    scatter!(p, [x2[end]], [y2[end]], [z2[end]], label="End 2", color=:pink, markersize=8)

    # Add center of mass
    scatter!(p, [0.0], [0.0], [0.0], label="Center of Mass", color=:black, markersize=6, markershape=:x)

    return p
end

# 3D Animation function
function create_3d_animation(sol, filename="orbital_motion_3d.gif")
    println("Creating 3D animation...")

    # Extract all 3D trajectory data
    x1 = [u[1] for u in sol.u]
    y1 = [u[2] for u in sol.u]
    z1 = [u[3] for u in sol.u]
    x2 = [u[4] for u in sol.u]
    y2 = [u[5] for u in sol.u]
    z2 = [u[6] for u in sol.u]

    # Determine plot bounds
    all_x = vcat(x1, x2)
    all_y = vcat(y1, y2)
    all_z = vcat(z1, z2)
    margin = 0.1 * max(maximum(all_x) - minimum(all_x),
        maximum(all_y) - minimum(all_y),
        maximum(all_z) - minimum(all_z))

    xlims = (minimum(all_x) - margin, maximum(all_x) + margin)
    ylims = (minimum(all_y) - margin, maximum(all_y) + margin)
    zlims = (minimum(all_z) - margin, maximum(all_z) + margin)

    # Create 3D animation
    anim = @animate for i in 1:length(sol.u)
        # Show trails (last 50 points)
        trail_start = max(1, i - 50)

        plot(
            title="3D Orbital Motion (t = $(round(sol.t[i], digits=1))s)",
            xlabel="X Position (m)", ylabel="Y Position (m)", zlabel="Z Position (m)",
            xlims=xlims, ylims=ylims, zlims=zlims,
            size=(800, 600),
            background_color=:black, foreground_color=:white,
            camera=(45, 30)  # Set viewing angle
        )

        # Plot 3D trails
        if i > 1
            plot!(x1[trail_start:i], y1[trail_start:i], z1[trail_start:i],
                color=:cyan, alpha=0.6, linewidth=2, label="")
            plot!(x2[trail_start:i], y2[trail_start:i], z2[trail_start:i],
                color=:orange, alpha=0.6, linewidth=2, label="")
        end

        # Plot current particles
        scatter!([x1[i]], [y1[i]], [z1[i]], color=:cyan, markersize=12,
            label="Massive Body", markerstrokewidth=2, markerstrokecolor=:white)
        scatter!([x2[i]], [y2[i]], [z2[i]], color=:orange, markersize=8,
            label="Smaller Body", markerstrokewidth=2, markerstrokecolor=:white)

        # Add center of mass
        scatter!([0], [0], [0], color=:yellow, markersize=4,
            label="Center of Mass", markershape=:star)
    end every 3  # Skip frames for faster creation

    # Save as GIF
    gif(anim, filename, fps=15)
    println("3D Animation saved as: $filename")

    return anim
end

# Test with interesting 3D initial conditions
function run_3d_inclined_orbit()
    println("Creating 3D simulation with inclined orbit...")

    # Parameters
    total_mass = 2e12
    m1, m2 = total_mass * 0.7, total_mass * 0.3
    separation = 2.0

    # Create an inclined orbit (30 degrees from XY plane)
    inclination = π / 6  # 30 degrees

    r1 = separation * m2 / (m1 + m2)
    r2 = separation * m1 / (m1 + m2)
    v_orbit = sqrt(G * (m1 + m2) / separation)

    # 3D orbital motion with inclination
    u0 = [
        -r1, 0.0, 0.0,                                    # Particle 1 position
        r2, 0.0, 0.0,                                    # Particle 2 position
        0.0, -v_orbit * m2 / (m1 + m2) * cos(inclination),  # Particle 1 velocity
        -v_orbit * m2 / (m1 + m2) * sin(inclination),
        0.0, v_orbit * m1 / (m1 + m2) * cos(inclination),  # Particle 2 velocity
        v_orbit * m1 / (m1 + m2) * sin(inclination)
    ]

    tspan = (0.0, 300.0)  # Longer simulation
    prob = ODEProblem(two_body_dynamics_3d!, u0, tspan, (m1, m2))
    sol = solve(prob, Vern7(), abstol=1e-12, reltol=1e-10, saveat=0.1)

    return sol
end

# Run the 3D simulation
println("Running 3D two-body gravitational simulation...")
@time solution_3d = run_two_body_simulation_3d()
println("3D Simulation complete!")

# Create 3D visualization
println("Creating 3D plot...")
plot_3d = plot_simulation_3d(solution_3d)
display(plot_3d)

# Create 3D animation
create_3d_animation(solution_3d)

# Try the inclined orbit version
println("\n=== Creating Inclined Orbit Demo ===")
inclined_solution = run_3d_inclined_orbit()
create_3d_animation(inclined_solution, "inclined_orbit_3d.gif")

println("✅ 3D simulation complete!")
println("Check the generated GIF files for beautiful 3D orbital motion!")