import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as patches

def visualize_basic_transformations():
    """Show basic 2D linear transformations"""
    
    # Original vectors to transform
    original_vectors = np.array([[1, 0], [0, 1], [1, 1], [-1, 1]]).T
    
    # Create a unit square for visualization
    square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    transformations = [
        ('Identity', np.array([[1, 0], [0, 1]])),
        ('Scale by 2', np.array([[2, 0], [0, 2]])),
        ('Horizontal Stretch', np.array([[2, 0], [0, 1]])),
        ('Rotation 45°', np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], 
                                   [np.sin(np.pi/4), np.cos(np.pi/4)]])),
        ('Shear', np.array([[1, 1], [0, 1]])),
        ('Reflection over x-axis', np.array([[1, 0], [0, -1]]))
    ]
    
    for idx, (name, matrix) in enumerate(transformations):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        # Transform the vectors
        transformed_vectors = matrix @ original_vectors
        transformed_square = matrix @ square
        
        # Plot original square (light gray)
        ax.plot(square[0], square[1], 'lightgray', linewidth=2, alpha=0.5, label='Original')
        ax.fill(square[0], square[1], 'lightgray', alpha=0.2)
        
        # Plot transformed square
        ax.plot(transformed_square[0], transformed_square[1], 'red', linewidth=2, label='Transformed')
        ax.fill(transformed_square[0], transformed_square[1], 'red', alpha=0.3)
        
        # Plot basis vectors
        colors = ['blue', 'green', 'purple', 'orange']
        labels = ['e₁', 'e₂', 'e₁+e₂', 'e₂-e₁']
        
        for i in range(len(original_vectors[0])):
            # Original vector
            ax.arrow(0, 0, original_vectors[0,i], original_vectors[1,i], 
                    head_width=0.1, head_length=0.1, fc='lightgray', ec='lightgray', alpha=0.5)
            
            # Transformed vector
            ax.arrow(0, 0, transformed_vectors[0,i], transformed_vectors[1,i],
                    head_width=0.1, head_length=0.1, fc=colors[i], ec=colors[i])
            
            # Label the transformed vector
            ax.text(transformed_vectors[0,i]*1.1, transformed_vectors[1,i]*1.1, 
                   f'T({labels[i]})', fontsize=10, color=colors[i])
        
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(f'{name}\\nMatrix: {matrix}', fontsize=12)
        if idx == 0:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.suptitle('Basic 2D Linear Transformations', fontsize=16, y=1.02)
    plt.show()

def show_matrix_as_transformation():
    """Demonstrate how to read a transformation matrix"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Example matrix
    A = np.array([[2, 1], [0, 1]])
    
    # Show what happens to basis vectors
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    Ae1 = A @ e1  # First column of A
    Ae2 = A @ e2  # Second column of A
    
    # Plot 1: Show basis vectors and their transformations
    ax = axes[0]
    
    # Original basis vectors
    ax.arrow(0, 0, e1[0], e1[1], head_width=0.1, head_length=0.1, 
             fc='blue', ec='blue', linewidth=2, label='e₁ = [1,0]')
    ax.arrow(0, 0, e2[0], e2[1], head_width=0.1, head_length=0.1, 
             fc='red', ec='red', linewidth=2, label='e₂ = [0,1]')
    
    # Transformed basis vectors
    ax.arrow(0, 0, Ae1[0], Ae1[1], head_width=0.1, head_length=0.1,
             fc='darkblue', ec='darkblue', linewidth=3, alpha=0.7, label='T(e₁) = [2,0]')
    ax.arrow(0, 0, Ae2[0], Ae2[1], head_width=0.1, head_length=0.1,
             fc='darkred', ec='darkred', linewidth=3, alpha=0.7, label='T(e₂) = [1,1]')
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title('Where Basis Vectors Go')
    
    # Plot 2: Show the matrix construction
    ax = axes[1]
    ax.text(0.5, 0.7, 'Matrix A = [T(e₁) | T(e₂)]', fontsize=14, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax.text(0.5, 0.5, f'A = [[2, 1],\\n     [0, 1]]', fontsize=16, ha='center', 
            fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    ax.text(0.5, 0.3, 'Column 1 = T(e₁) = [2, 0]\\nColumn 2 = T(e₂) = [1, 1]', 
            fontsize=12, ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Reading the Matrix')
    
    # Plot 3: Show transformation of arbitrary vector
    ax = axes[2]
    
    # Arbitrary vector
    v = np.array([1, 1])
    Av = A @ v
    
    # Show decomposition: v = 1*e1 + 1*e2
    ax.arrow(0, 0, e1[0], e1[1], head_width=0.05, head_length=0.05,
             fc='blue', ec='blue', alpha=0.5, linewidth=1)
    ax.arrow(e1[0], e1[1], e2[0], e2[1], head_width=0.05, head_length=0.05,
             fc='red', ec='red', alpha=0.5, linewidth=1)
    ax.arrow(0, 0, v[0], v[1], head_width=0.08, head_length=0.08,
             fc='purple', ec='purple', linewidth=2, label='v = [1,1]')
    
    # Show transformed vector: T(v) = 1*T(e1) + 1*T(e2)
    ax.arrow(0, 0, Ae1[0], Ae1[1], head_width=0.05, head_length=0.05,
             fc='darkblue', ec='darkblue', alpha=0.7, linewidth=1)
    ax.arrow(Ae1[0], Ae1[1], Ae2[0], Ae2[1], head_width=0.05, head_length=0.05,
             fc='darkred', ec='darkred', alpha=0.7, linewidth=1)
    ax.arrow(0, 0, Av[0], Av[1], head_width=0.08, head_length=0.08,
             fc='darkmagenta', ec='darkmagenta', linewidth=3, label='T(v) = [3,1]')
    
    ax.text(0.5, 0.5, 'v', fontsize=12, color='purple')
    ax.text(Av[0]-0.2, Av[1], 'T(v)', fontsize=12, color='darkmagenta')
    
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 1.5)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_title('Transforming Any Vector')
    
    plt.tight_layout()
    plt.show()

def demonstrate_composition():
    """Show how matrix multiplication represents composition of transformations"""
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Define two transformations
    # First: rotate by 45 degrees
    theta = np.pi/4
    R = np.array([[np.cos(theta), -np.sin(theta)], 
                  [np.sin(theta), np.cos(theta)]])
    
    # Second: scale by [2, 1]
    S = np.array([[2, 0], [0, 1]])
    
    # Composition: S ∘ R (first rotate, then scale)
    composition = S @ R
    
    # Test vector
    v = np.array([1, 0])
    
    # Apply transformations step by step
    step1 = R @ v      # First rotate
    step2 = S @ step1  # Then scale
    direct = composition @ v  # Direct composition
    
    transformations = [
        ('Original', np.eye(2), v, 'blue'),
        ('After Rotation', R, step1, 'red'),
        ('After Scaling', S, step2, 'green'),
        ('Direct Composition', composition, direct, 'purple')
    ]
    
    for idx, (title, matrix, result, color) in enumerate(transformations):
        ax = axes[idx]
        
        # Draw unit square
        square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]])
        if idx > 0:
            transformed_square = matrix @ square
            ax.plot(transformed_square[0], transformed_square[1], color, linewidth=2)
            ax.fill(transformed_square[0], transformed_square[1], color, alpha=0.3)
        else:
            ax.plot(square[0], square[1], color, linewidth=2)
            ax.fill(square[0], square[1], color, alpha=0.3)
        
        # Draw the result vector
        ax.arrow(0, 0, result[0], result[1], head_width=0.1, head_length=0.1,
                fc=color, ec=color, linewidth=3)
        ax.text(result[0]*1.1, result[1]*1.1, f'[{result[0]:.1f}, {result[1]:.1f}]',
                fontsize=10, color=color)
        
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.set_title(f'{title}\\n{matrix}')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Step by step: [1,0] → R → {step1} → S → {step2}")
    print(f"Direct: [1,0] → (S∘R) → {direct}")
    print(f"Matrix multiplication: S @ R = \\n{composition}")

def show_geometric_intuition():
    """Show the geometric intuition behind linear transformations"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Create a grid of points
    x = np.linspace(-2, 2, 9)
    y = np.linspace(-2, 2, 9)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()])
    
    # Transformation matrix (shear)
    A = np.array([[1, 0.5], [0.5, 1]])
    
    # Transform all points
    transformed_points = A @ points
    transformed_X = transformed_points[0].reshape(X.shape)
    transformed_Y = transformed_points[1].reshape(Y.shape)
    
    # Plot original grid
    ax1.plot(X, Y, 'b-', alpha=0.5)
    ax1.plot(X.T, Y.T, 'b-', alpha=0.5)
    ax1.scatter(X, Y, c='blue', s=20, alpha=0.7)
    
    # Highlight some special vectors
    origin = np.array([0, 0])
    e1 = np.array([1, 0])
    e2 = np.array([0, 1])
    
    ax1.arrow(0, 0, e1[0], e1[1], head_width=0.1, head_length=0.1,
             fc='red', ec='red', linewidth=3, label='e₁')
    ax1.arrow(0, 0, e2[0], e2[1], head_width=0.1, head_length=0.1,
             fc='green', ec='green', linewidth=3, label='e₂')
    
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Before Transformation\\n(Original Grid)')
    ax1.legend()
    
    # Plot transformed grid
    ax2.plot(transformed_X, transformed_Y, 'r-', alpha=0.5)
    ax2.plot(transformed_X.T, transformed_Y.T, 'r-', alpha=0.5)
    ax2.scatter(transformed_X, transformed_Y, c='red', s=20, alpha=0.7)
    
    # Show where basis vectors go
    Ae1 = A @ e1
    Ae2 = A @ e2
    
    ax2.arrow(0, 0, Ae1[0], Ae1[1], head_width=0.1, head_length=0.1,
             fc='darkred', ec='darkred', linewidth=3, label='T(e₁)')
    ax2.arrow(0, 0, Ae2[0], Ae2[1], head_width=0.1, head_length=0.1,
             fc='darkgreen', ec='darkgreen', linewidth=3, label='T(e₂)')
    
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title('After Transformation\\n(Deformed Grid)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Key insights:")
    print("• Straight lines remain straight")
    print("• Parallel lines remain parallel") 
    print("• The origin stays fixed")
    print("• The grid deforms uniformly")
    print(f"• Basis vectors: e₁ → {Ae1}, e₂ → {Ae2}")

# Run the demonstrations
if __name__ == "__main__":
    print("1. Basic Linear Transformations")
    visualize_basic_transformations()
    
    print("\\n2. How to Read a Transformation Matrix")
    show_matrix_as_transformation()
    
    print("\\n3. Composition of Transformations")
    demonstrate_composition()
    
    print("\\n4. Geometric Intuition - Grid Deformation")
    show_geometric_intuition()