clc; clear; close all;

% Franke Function Approximation using Halton sequence for RBF centers
runExperiment();

function runExperiment()
    % Parameters
    sample_size = 50; % Number of RBF centers
    sigma_values = [0.05, 0.1, 0.25, 0.5]; % Different sigma values for comparison
    
    % Generate Halton sequence for RBF centers
    [X_centers, y_centers] = haltonSampling(sample_size);
    
    % Create dense test grid
    [x_test, y_test] = meshgrid(linspace(0, 1, 100), linspace(0, 1, 100));
    X_test = [x_test(:), y_test(:)];
    
    % True values
    y_true = arrayfun(@(a, b) frankeFunction(a, b), x_test(:), y_test(:));
    
    % Prepare figures for approximation and error
    figure('Name', 'Franke Function Approximations');
    figure('Name', 'L2 Error Distributions');
    
    % Experiment with different sigma values
    for i = 1:length(sigma_values)
        sigma = sigma_values(i);
        
        % Interpolation
        y_approx = rbfInterpolation(X_centers, y_centers, X_test, sigma);
        
        % Compute errors
        error = abs(y_true - y_approx);
        l2_error_grid = reshape(error, size(x_test)); % Reshape error to grid dimensions
        approx_grid = reshape(y_approx, size(x_test)); % Reshape approximation to grid dimensions
        

        % Compute RMSE and Max Error
        RMSE = sqrt(mean(error.^2)); % Root Mean Squared Error
        Max_Error = max(error); % Maximum Error
        
        % Display the errors for each sigma
        disp(['Sigma = ', num2str(sigma), ':']);
        disp(['  RMSE: ', num2str(RMSE)]);
        disp(['  Max Error: ', num2str(Max_Error)]);

        % Visualization of approximation
        figure(1);
        subplot(2, 2, i);
        contourf(x_test, y_test, approx_grid, 20, 'LineStyle', 'none');
        title(['Sigma = ', num2str(sigma), ' (Approximation)']);
        xlabel('x'); ylabel('y'); colorbar;
        
        % Visualization of error
        figure(2);
        subplot(2, 2, i);
        contourf(x_test, y_test, l2_error_grid, 20, 'LineStyle', 'none');
        title(['Sigma = ', num2str(sigma), ' (L2 Error)']);
        xlabel('x'); ylabel('y'); colorbar;

    end
    figure(1);
    sgtitle(['Franke Function Approximation with Halton Sampling']);
    
    figure(2);
    sgtitle(['L2 Error Distribution with Halton Sampling']);
end

function z = frankeFunction(x, y)
    % Implementation of the Franke test function
    term1 = 0.75 * exp(-((9 * x - 2).^2 + (9 * y - 2).^2) / 4);
    term2 = 0.75 * exp(-((9 * x + 1).^2) / 49 - (9 * y + 1) / 10);
    term3 = 0.5 * exp(-((9 * x - 7).^2 + (9 * y - 3).^2) / 4);
    term4 = -0.2 * exp(-(9 * x - 4).^2 - (9 * y - 7).^2);
    z = term1 + term2 + term3 + term4;
end

function [interpolant] = rbfInterpolation(X_centers, y_centers, X_test, sigma)
    % Implementation of RBF Interpolation
    % RBF kernel: Gaussian kernel with parameter sigma
    N = size(X_centers, 1);
    M = size(X_test, 1);
    
    % Compute weights (solve linear system)
    K_centers = zeros(N, N);
    for i = 1:N
        for j = 1:N
            K_centers(i, j) = exp(-norm(X_centers(i, :) - X_centers(j, :))^2 / (2 * sigma^2));
        end
    end
    weights = K_centers \ y_centers; % Solve for weights
    
    % Compute interpolation at test points
    interpolant = zeros(M, 1);
    for i = 1:M
        for j = 1:N
            interpolant(i) = interpolant(i) + weights(j) * exp(-norm(X_test(i, :) - X_centers(j, :))^2 / (2 * sigma^2));
        end
    end
end

function [X, y] = haltonSampling(n)
    % Generate Halton sequence for sampling
    p1 = haltonset(2, 'Skip', 1e3, 'Leap', 1e2);
    X = net(p1, n);
    y = arrayfun(@(a, b) frankeFunction(a, b), X(:, 1), X(:, 2));
end