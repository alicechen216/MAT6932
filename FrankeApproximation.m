clc; clear; close all;

% Franke Function Approximation using RBF Interpolation
% Experiment with different sigma values
runExperiment();

function runExperiment()
    % Parameters
    sampling_method = 'lattice'; % Choose 'lattice' or 'halton'
    sample_size = 50; % Number of training points
    sigma_values = [0.05, 0.1, 0.25, 0.5]; % Different sigma values for comparison

    [X_train, y_train] = latticeSampling(sample_size);

    % Create dense test grid
    [x_test, y_test] = meshgrid(linspace(0, 1, 100), linspace(0, 1, 100));
    X_test = [x_test(:), y_test(:)];
    
    % True values
    y_true = arrayfun(@(a, b) frankeFunction(a, b), x_test(:), y_test(:));
    
    % Prepare figures for visualization
    figure('Name', 'Franke Function Approximation');
    figure('Name', 'L2 Error Visualization');
    
    % Experiment with different sigma values
    for i = 1:length(sigma_values)
        sigma = sigma_values(i);
        
        % Interpolation
        y_approx = rbfInterpolation(X_train, y_train, X_test, sigma);
        
        % Compute errors
        rmse = sqrt(mean((y_true - y_approx).^2));
        maxError = max(abs(y_true - y_approx));
        l2_error = reshape(abs(y_true - y_approx), size(x_test)); % Reshape L2 error for plotting
        
        % Display results
        fprintf('Sigma = %.3f:\n', sigma);
        fprintf('  RMSE: %.4f\n', rmse);
        fprintf('  Max Error: %.4f\n\n', maxError);
        
        % Visualization of approximation
        figure(1);
        subplot(2, 2, i);
        scatter(X_test(:, 1), X_test(:, 2), 50, y_approx, 'filled');
        title(['Sigma = ', num2str(sigma), ' (Approximation)']);
        xlabel('x'); ylabel('y'); colorbar;
        
        % Visualization of L2 error
        figure(2);
        subplot(2, 2, i);
        contourf(x_test, y_test, l2_error, 20, 'LineStyle', 'none');
        title(['Sigma = ', num2str(sigma), ' (L2 Error)']);
        xlabel('x'); ylabel('y'); colorbar;
    end
    
    % Add overall titles
    figure(1);
    sgtitle(['Franke Function Approximation with ', sampling_method, ' Sampling']);
    
    figure(2);
    sgtitle(['L2 Error Distribution with ', sampling_method, ' Sampling']);
end

function z = frankeFunction(x, y)
    % Implementation of the Franke test function
    term1 = 0.75 * exp(-((9 * x - 2).^2 + (9 * y - 2).^2) / 4);
    term2 = 0.75 * exp(-((9 * x + 1).^2) / 49 - (9 * y + 1) / 10);
    term3 = 0.5 * exp(-((9 * x - 7).^2 + (9 * y - 3).^2) / 4);
    term4 = -0.2 * exp(-(9 * x - 4).^2 - (9 * y - 7).^2);
    z = term1 + term2 + term3 + term4;
end

function [interpolant] = rbfInterpolation(X_train, y_train, X_test, sigma)
    % Implementation of RBF Interpolation
    % RBF kernel: Gaussian kernel with parameter sigma
    N = size(X_train, 1);
    M = size(X_test, 1);
    
    % Compute weights (solve linear system)
    K_train = zeros(N, N);
    for i = 1:N
        for j = 1:N
            K_train(i, j) = exp(-norm(X_train(i, :) - X_train(j, :))^2 / (2 * sigma^2));
        end
    end
    weights = K_train \ y_train; % Solve for weights
    
    % Compute interpolation at test points
    interpolant = zeros(M, 1);
    for i = 1:M
        for j = 1:N
            interpolant(i) = interpolant(i) + weights(j) * exp(-norm(X_test(i, :) - X_train(j, :))^2 / (2 * sigma^2));
        end
    end
end

function [X, y] = latticeSampling(n)
    % Generate lattice grid sampling
    [x, y] = meshgrid(linspace(0, 1, sqrt(n)), linspace(0, 1, sqrt(n)));
    X = [x(:), y(:)];
    y = arrayfun(@(a, b) frankeFunction(a, b), x(:), y(:));
end