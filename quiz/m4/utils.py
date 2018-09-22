import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def create_corr_data(corr = 0.8, data_min = 10, data_max = 80):
    
    # Set the Random Seed
    np.random.seed(seed = 0)
    
    # Set the x and y range for the data
    data_x_range = np.array([data_min, data_max])
    data_y_range = np.array([data_min, data_max])

    # Set the average of the x and y postions of the data points
    means = [data_x_range.mean(), data_y_range.mean()]
    
    # Set the standard deviation of the x and y positions of the data points
    stds = [data_x_range.std() / 3, data_y_range.std() / 3]

    # Construct the Covariance matrix
    covs = [[stds[0]**2, stds[0]*stds[1]*corr], 
            [stds[0]*stds[1]*corr, stds[1]**2]] 

    # Return the correlated data with 1000 data points
    return np.random.multivariate_normal(means, covs, 1000)


def plot_data(X, corr=None):
    
    # Make a Scatter plot of the data
    plt.scatter(X[:,0], X[:,1],color = 'white', alpha = 0.5, linewidth = 0)
    ax = plt.gca()
    ax.set_facecolor('lightslategray')
    plt.grid()
    plt.xlabel('$X$', fontsize = 20)
    plt.ylabel('$Y$', fontsize = 20)
    
    # Put legend on plot if correlation is provided
    if corr:
        props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)
        textstr = 'Corr(X,Y) = %.2f'%(corr)
        plt.text(0.05, 0.91, textstr, fontsize = 20, horizontalalignment = 'left',
                verticalalignment = 'top', bbox = props, transform = ax.transAxes)
        
    plt.show()
    
    
def mean_normalize_data(X):
    
    # Average of the values in each column of X
    ave_cols = X.mean(axis = 0)

    # Standard Deviation of the values in each column of X
    std_cols = X.std(axis = 0)

    # Return the mean normalized X
    return (X - ave_cols) / std_cols


def rotate_data(X, angle = 45.0):
    
    # Convert the angle from degrees to radians
    theta = (angle / 180.0) * np.pi
    
    #Create the 2D rotation matrix
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])

    # Return the rotated X
    return np.matmul(X, rotMatrix)


def get_pca_comp(X, n_cpm = 2):
    
    # Perform PCA for the given number of components
    pca = PCA(n_components = n_cpm)
    pca.fit(X)
    
    # Return the principal components
    return pca.components_


def get_proj_pts(X, pca_components):

    # Rotate X to get the min and max values
    X_rot = rotate_data(X)

    # Create and empty array to hold our projected points
    proj_pts = np.array([])

    for i in range(2):
        
        # Get the indices of the min and max data points
        min_idx = X_rot[:,i].argmin()
        max_idx = X_rot[:,i].argmax()

        # Select the x and y coordinates of the points to be projected
        q1 = np.array([X[min_idx,0], X[min_idx,1]])
        q2 = np.array([X[max_idx,0], X[max_idx,1]])

        # Set a point equal to the origin
        p0 = np.array([0,0])

        # Set a point that is in the direction of the ith eigenvector
        p1 = pca_components[i]

        # Create the A matrices
        a1 = np.array([[-q1[0]*(p1[0]-p0[0]) - q1[1]*(p1[1]-p0[1])], [-p0[1]*(p1[0]-p0[0]) + p0[0]*(p1[1]-p0[1])]])
        a2 = np.array([[-q2[0]*(p1[0]-p0[0]) - q2[1]*(p1[1]-p0[1])], [-p0[1]*(p1[0]-p0[0]) + p0[0]*(p1[1]-p0[1])]])

        # Create the B matrix
        b = np.array([[p1[0] - p0[0], p1[1] - p0[1]], [p0[1] - p1[1], p1[0] - p0[0]]])

        # Solve the linear equations Bx = A
        proj_pt1 = - np.linalg.solve(b, a1)
        proj_pt2 = - np.linalg.solve(b, a2)

        # Append the values of the projected points
        proj_pts = np.append(proj_pts, proj_pt1)
        proj_pts = np.append(proj_pts, proj_pt2)
        
    return proj_pts


def plot_data_with_pca_comp(corr = 0.8, data_min = 10, data_max = 80):
    
    # Create correlated data
    X = create_corr_data(corr, data_min, data_max)

    # Mean normalize X
    X_norm = mean_normalize_data(X)

    # Get the principal components    
    pca_components = get_pca_comp(X_norm)

    # Get the projected points onto the principal components
    proj_pts = get_proj_pts(X_norm, pca_components)

    # Set the first and second components
    first_pc = pca_components[0]
    second_pc = pca_components[1]

    # Plot the correlated data
    plt.scatter(X_norm[:,0], X_norm[:,1], color = 'white', alpha = 0.5, linewidth = 0)
    ax = plt.gca()
    ax.set_facecolor('lightslategray')
    plt.grid()

    # Plot the first principal component (eigenvector) as a line
    x_vals = np.linspace(proj_pts[1], proj_pts[3], 10)
    y_vals = (first_pc[1] / first_pc[0]) * x_vals
    plt.plot(x_vals, y_vals, color = 'm', linewidth = 2, label = '$1^{st}$ PC')

    # Plot the second principal component (eigenvector) as a line
    x_vals = np.linspace(-proj_pts[5], -proj_pts[7], 10)
    y_vals = (second_pc[1] / second_pc[0]) * x_vals
    plt.plot(x_vals, y_vals, color = 'c', linewidth = 2, label = '$2^{nd}$ PC')
    plt.legend()

#     # Plot projected points
#     plt.scatter(proj_pts[0],proj_pts[1], color='r')
#     plt.scatter(proj_pts[2],proj_pts[3], color='c')
#     plt.scatter(proj_pts[4],proj_pts[5], color='r')
#     plt.scatter(proj_pts[6],proj_pts[7], color='c')

    # Plot legend
    props = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5)
    textstr = 'Corr(X,Y) = %.2f'%(corr)
    plt.text(0.05, 0.91, textstr, fontsize = 20, horizontalalignment = 'left',
            verticalalignment = 'top', bbox = props, transform = ax.transAxes)
    
    # Set the axis to equal or square
    plt.axis('equal')
#     plt.axis('square')

    plt.show()