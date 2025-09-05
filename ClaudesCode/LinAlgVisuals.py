import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up the plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 12)

def plot_2d_column_independence():
    """Visualize column independence vs dependence in 2D"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Example 1: Independent columns
    v1 = np.array([1, 0])  # East
    v2 = np.array([0, 1])  # North
    
    ax1.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='Column 1: [1,0]')
    ax1.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, label='Column 2: [0,1]')
    
    # Show what we can reach (span)
    x_span = np.linspace(-2, 2, 100)
    y_span = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x_span, y_span)
    ax1.contour(X, Y, X**2 + Y**2, levels=[1, 4], colors='lightgray', alpha=0.3)
    ax1.fill_between([-2, 2], [-2, -2], [2, 2], alpha=0.1, color='green', label='Span: All of R²')
    
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title('Independent Columns: Can reach anywhere in 2D')
    
    # Example 2: Dependent columns
    v1_dep = np.array([1, 2])    # Northeast
    v2_dep = np.array([2, 4])    # Same direction, scaled
    
    ax2.quiver(0, 0, v1_dep[0], v1_dep[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='Column 1: [1,2]')
    ax2.quiver(0, 0, v2_dep[0], v2_dep[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, label='Column 2: [2,4]')
    
    # Show the line we're constrained to (y = 2x)
    x_line = np.linspace(-2.5, 2.5, 100)
    y_line = 2 * x_line
    ax2.plot(x_line, y_line, 'g-', linewidth=3, alpha=0.7, label='Span: Line y=2x')
    
    # Show some example reachable points
    example_points = np.array([[0.5, 1], [1, 2], [-1, -2], [1.5, 3]])
    ax2.scatter(example_points[:, 0], example_points[:, 1], color='green', s=50, zorder=5)
    
    # Show an unreachable point
    ax2.scatter([1], [1], color='red', s=100, marker='x', zorder=5, label='Unreachable: [1,1]')
    
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title('Dependent Columns: Constrained to line')
    
    plt.tight_layout()
    plt.show()

def plot_3d_column_dependence():
    """Visualize 3D column dependence - constrained to 2D plane"""
    fig = plt.figure(figsize=(15, 5))
    
    # Independent case
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Independent vectors
    v1 = np.array([1, 0, 0])  # x-direction
    v2 = np.array([0, 1, 0])  # y-direction  
    v3 = np.array([0, 0, 1])  # z-direction
    
    ax1.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='red', label='[1,0,0]')
    ax1.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='blue', label='[0,1,0]')
    ax1.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='green', label='[0,0,1]')
    
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(0, 1.5)
    ax1.set_zlim(0, 1.5)
    ax1.set_title('Independent: Span all of R³')
    ax1.legend()
    
    # Dependent case - constrained to plane
    ax2 = fig.add_subplot(132, projection='3d')
    
    v1_dep = np.array([1, 0, 1])  # diagonal in xz-plane
    v2_dep = np.array([0, 1, 0])  # y-direction
    v3_dep = np.array([2, 0, 2])  # same as v1, scaled
    
    ax2.quiver(0, 0, 0, v1_dep[0], v1_dep[1], v1_dep[2], color='red', label='[1,0,1]')
    ax2.quiver(0, 0, 0, v2_dep[0], v2_dep[1], v2_dep[2], color='blue', label='[0,1,0]')
    ax2.quiver(0, 0, 0, v3_dep[0], v3_dep[1], v3_dep[2], color='orange', label='[2,0,2]', alpha=0.7)
    
    # Show the plane x = z
    xx, yy = np.meshgrid(np.linspace(-1, 3, 10), np.linspace(-2, 2, 10))
    zz = xx  # This makes z = x (the constraint plane)
    ax2.plot_surface(xx, yy, zz, alpha=0.3, color='yellow')
    
    ax2.set_xlim(-1, 3)
    ax2.set_ylim(-2, 2)
    ax2.set_zlim(-1, 3)
    ax2.set_title('Dependent: Span is plane x=z')
    ax2.legend()
    
    # Show solution examples on the constraint plane
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Same vectors
    ax3.quiver(0, 0, 0, v1_dep[0], v1_dep[1], v1_dep[2], color='red', label='[1,0,1]')
    ax3.quiver(0, 0, 0, v2_dep[0], v2_dep[1], v2_dep[2], color='blue', label='[0,1,0]')
    ax3.plot_surface(xx, yy, zz, alpha=0.2, color='yellow')
    
    # Show example solutions on the plane
    solution_points = np.array([[1, 0, 1], [2, 1, 2], [0, -1, 0], [1.5, 2, 1.5]])
    ax3.scatter(solution_points[:, 0], solution_points[:, 1], solution_points[:, 2], 
               color='green', s=100, label='Reachable points')
    
    # Show unreachable point (not on plane x=z)
    ax3.scatter([1], [0], [2], color='red', s=100, marker='x', label='Unreachable [1,0,2]')
    
    ax3.set_xlim(-1, 3)
    ax3.set_ylim(-2, 2)
    ax3.set_zlim(-1, 3)
    ax3.set_title('Solution Points on Constraint Plane')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_system_solutions():
    """Show different solution types geometrically"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Unique solution (independent columns)
    ax = axes[0, 0]
    x = np.linspace(-1, 4, 100)
    y1 = (3 - x) / 2      # From x + 2y = 3
    y2 = 3 - x            # From x + y = 3
    
    ax.plot(x, y1, 'r-', label='x + 2y = 3', linewidth=2)
    ax.plot(x, y2, 'b-', label='x + y = 3', linewidth=2)
    ax.scatter([3], [0], color='green', s=100, zorder=5, label='Solution: (3,0)')
    ax.set_xlim(-0.5, 4)
    ax.set_ylim(-0.5, 4)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Unique Solution\n(Independent columns)')
    
    # 2. Infinitely many solutions (dependent columns)
    ax = axes[0, 1]
    x = np.linspace(-1, 4, 100)
    y = (3 - x) / 2       # Both equations give same line
    
    ax.plot(x, y, 'purple', linewidth=4, alpha=0.7, label='x + 2y = 3 (both equations)')
    # Show some solution points
    sol_x = np.array([1, 3, -1, 5])
    sol_y = (3 - sol_x) / 2
    ax.scatter(sol_x, sol_y, color='green', s=60, zorder=5, label='Solution points')
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 3)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Infinitely Many Solutions\n(Dependent columns)')
    
    # 3. No solution (inconsistent)
    ax = axes[1, 0]
    x = np.linspace(-1, 5, 100)
    y1 = (3 - x) / 2      # From x + 2y = 3
    y2 = (5 - x) / 2      # From x + 2y = 5 (parallel line)
    
    ax.plot(x, y1, 'r-', label='x + 2y = 3', linewidth=2)
    ax.plot(x, y2, 'b-', label='x + 2y = 5', linewidth=2)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 3)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('No Solution\n(Inconsistent system)')
    
    # 4. 3D plane intersections
    ax = fig.add_subplot(224, projection='3d')
    
    # Create three planes that intersect at a point
    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    X, Y = np.meshgrid(x, y)
    
    # x + y + z = 3
    Z1 = 3 - X - Y
    ax.plot_surface(X, Y, Z1, alpha=0.3, color='red', label='x + y + z = 3')
    
    # x - y + z = 1  
    Z2 = 1 - X + Y
    ax.plot_surface(X, Y, Z2, alpha=0.3, color='blue', label='x - y + z = 1')
    
    # 2x + z = 2
    Z3 = 2 - 2*X + 0*Y
    ax.plot_surface(X, Y, Z3, alpha=0.3, color='green', label='2x + z = 2')
    
    # Mark the intersection point (solution)
    ax.scatter([0], [1], [2], color='black', s=100, label='Solution: (0,1,2)')
    
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2) 
    ax.set_zlim(-2, 4)
    ax.set_title('3D: Three Planes Intersecting\nat Point (Unique Solution)')
    
    plt.tight_layout()
    plt.show()

def demonstrate_homogeneous_vs_nonhomogeneous():
    """Show the relationship between homogeneous and non-homogeneous solutions"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Homogeneous system: x + 2y = 0
    x = np.linspace(-4, 4, 100)
    y_homo = -x / 2
    
    ax1.plot(x, y_homo, 'b-', linewidth=3, label='x + 2y = 0')
    ax1.scatter([0], [0], color='red', s=100, zorder=5, label='Origin (trivial solution)')
    
    # Show some non-trivial solutions
    homo_x = np.array([-2, -1, 1, 2])
    homo_y = -homo_x / 2
    ax1.scatter(homo_x, homo_y, color='blue', s=60, alpha=0.7, label='Non-trivial solutions')
    
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linewidth=0.5)
    ax1.axvline(x=0, color='k', linewidth=0.5)
    ax1.legend()
    ax1.set_title('Homogeneous: x + 2y = 0\n(Always passes through origin)')
    
    # Non-homogeneous system: x + 2y = 3
    y_nonhomo = (3 - x) / 2
    
    ax2.plot(x, y_nonhomo, 'r-', linewidth=3, label='x + 2y = 3')
    ax2.plot(x, y_homo, 'b--', linewidth=2, alpha=0.5, label='x + 2y = 0 (homogeneous)')
    
    # Particular solution
    ax2.scatter([3], [0], color='green', s=100, zorder=5, label='Particular solution: (3,0)')
    
    # Show some other solutions
    nonhomo_x = np.array([1, -1, 5])
    nonhomo_y = (3 - nonhomo_x) / 2
    ax2.scatter(nonhomo_x, nonhomo_y, color='red', s=60, alpha=0.7, label='Other solutions')
    
    # Draw arrow showing the "shift"
    ax2.annotate('', xy=(3, 0), xytext=(0, 0), 
                arrowprops=dict(arrowstyle='->', lw=2, color='purple'))
    ax2.text(1.5, -0.5, 'Shift by\nparticular\nsolution', ha='center', color='purple')
    
    ax2.set_xlim(-2, 6)
    ax2.set_ylim(-2, 3)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linewidth=0.5)
    ax2.axvline(x=0, color='k', linewidth=0.5)
    ax2.legend()
    ax2.set_title('Non-homogeneous: x + 2y = 3\n(Parallel to homogeneous)')
    
    plt.tight_layout()
    plt.show()

# Run all visualizations
if __name__ == "__main__":
    print("1. Column Independence vs Dependence in 2D")
    plot_2d_column_independence()
    
    print("\n2. 3D Column Dependence - Constraint to 2D Plane")
    plot_3d_column_dependence()
    
    print("\n3. Different Solution Types")
    visualize_system_solutions()
    
    print("\n4. Homogeneous vs Non-homogeneous Solutions")
    demonstrate_homogeneous_vs_nonhomogeneous()