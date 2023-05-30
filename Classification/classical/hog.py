import numpy as np

def calc_gradient(image):
    # Get the gradient of the image in both directions using simple 1-d masks
    # [1, 0,-1] and [1, 0,-1].T 
    horizontal_grad = np.zeros(image.shape)
    horizontal_grad[1:-1, :] = image[2:, :] - image[:-2, :]
    vertical_grad = np.zeros(image.shape)
    vertical_grad[:, 1:-1] = image[:, 2:] - image[:, :-2]
    return horizontal_grad, vertical_grad
    
def cell_vote(magnitude, phase, orientation_start, orientation_end, cell_columns, cell_rows, column_index, row_index, size_columns, size_rows, range_rows_start, range_rows_stop, range_columns_start, range_columns_stop):
    total = 0.0    
    for cell_row in range(int(range_rows_start + row_index), int(range_rows_stop + row_index)):
        for cell_column in range(int(range_columns_start + column_index), int(range_columns_stop + column_index)):
            # Check if the phase is in range of the orientation_start and orientation_end
            if (orientation_end <= phase[cell_row, cell_column] < orientation_start):
                # We use the magnitude of the gradient as the vote(as in paper)
                # Add the magnitude to the total for the cell
                total += magnitude[cell_row, cell_column]
    # Return the average total for the cell 
    return total / (cell_rows * cell_columns)

def calculate_hog(magnitude, phase, pixels_per_col_cell, pixels_per_row_cell, number_of_columns, number_of_rows, number_of_cells_per_col, number_of_cells_per_row, orientations, orientation_histogram):    
    # Number of pixels in row and column of cells
    total_number_of_pixels_in_row = pixels_per_row_cell * number_of_cells_per_row
    total_number_of_pixels_in_col = pixels_per_col_cell * number_of_cells_per_col
    
    # Value added to row or column index (which is at the center of the cell) to get the range of pixels in the cell
    range_rows_stop = int((pixels_per_row_cell + 1) / 2)
    range_rows_start = int(-(pixels_per_row_cell / 2))
    range_columns_stop = int((pixels_per_col_cell + 1) / 2)
    range_columns_start = int(-(pixels_per_col_cell / 2))
    
    # Number of orientations per 180 degrees
    number_of_orientations_per_180 = 180. / orientations

    # compute orientations integral images
    for orientation in range(orientations):
        # isolate orientations in this range (0, 20), (20, 40), ... 
        orientation_upper_limit = number_of_orientations_per_180 * (orientation + 1)
        orientation_lower_limit = number_of_orientations_per_180 * orientation
        # Reset the cell row to 0
        cell_row = 0
        # Loop through the pixels from the center of the first cell to the center of the last cell
        for pixel_row in range(int(pixels_per_row_cell / 2), total_number_of_pixels_in_row, pixels_per_row_cell):
            # Reset the cell column to 0
            cell_column = 0
            # While c is within the image
            for pixel_column in range(int(pixels_per_col_cell / 2), total_number_of_pixels_in_col, pixels_per_col_cell):
                orientation_histogram[cell_row, cell_column, orientation] = cell_vote(magnitude, phase, orientation_upper_limit, orientation_lower_limit, pixels_per_col_cell, pixels_per_row_cell, pixel_column, pixel_row, number_of_columns, number_of_rows, range_rows_start, range_rows_stop, range_columns_start, range_columns_stop)
                cell_column += 1
            cell_row += 1

def get_features_hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    # Gradient of rows and columns (Ignore the last row and column)
    horizontal_grad, vertical_grad = calc_gradient(image)
    # Size of the image
    number_of_rows, number_of_columns = image.shape
    # Get magnitude and direction from vertical_grad and gradient_rows
    magnitude = np.sqrt(vertical_grad ** 2 + horizontal_grad ** 2)
    angle = np.rad2deg(np.arctan2(horizontal_grad, vertical_grad)) % 180
    # Create histogram matrix to store values with size of number of cells in rows and columns and number of orientations
    cell_histograms = np.zeros((int(number_of_rows / pixels_per_cell[0]), int(number_of_columns/ pixels_per_cell[1]), orientations))
    calculate_hog(magnitude, angle, pixels_per_cell[1], pixels_per_cell[0], number_of_columns, number_of_rows, int(number_of_columns/ pixels_per_cell[1]), int(number_of_rows / pixels_per_cell[0]), orientations, cell_histograms)
    block_matrix = np.zeros((int(number_of_rows / pixels_per_cell[0]) - cells_per_block[0] + 1, int(number_of_columns / pixels_per_cell[1]) - cells_per_block[1] + 1, cells_per_block[0] * cells_per_block[1] * orientations))
    for i in range(block_matrix.shape[0]):
        for j in range(block_matrix.shape[1]):
            block_matrix[i, j] = cell_histograms[i:i + cells_per_block[0], j:j + cells_per_block[1]].flatten()
            # L2 normlization then clipping to 0.2 and renormalization
            eps = 1e-5
            block_matrix[i, j] = block_matrix[i, j] / np.sqrt(np.sum(block_matrix[i, j] ** 2) + eps ** 2)
            block_matrix[i, j][block_matrix[i, j] >= 0.2] = 0.2
            block_matrix[i, j] = block_matrix[i, j] / np.sqrt(np.sum(block_matrix[i, j] ** 2) + eps ** 2)
    
    block_matrix = block_matrix.flatten()
    return block_matrix