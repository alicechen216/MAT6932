clc; clear; close all;
% Tent Map RBF Approximation Project
% Comprehensive script for approximating the Tent Map using different kernel methods
% 
% Set up experiment parameters
initial_conditions = [0.1, 0.5, 0.7, 0.9];
n_iterations = 100;
sigmas = [0.1, 0.5, 1, 2];

% Run the experiment
results = runTentMapExperiment(initial_conditions, n_iterations, sigmas);

% Visualize results
visualizeResults(results);

% Print detailed results
disp('Approximation Results:');
for i = 1:length(results)
    fprintf('Initial Condition: %.2f, Sigma: %.2f\n', ...
        results(i).initial_condition, results(i).sigma);
    fprintf('  Gaussian Kernel Error: %e\n', results(i).gaussian_error);
    fprintf('  Polynomial Kernel Error: %e\n', results(i).polynomial_error);
    fprintf('  Exponential Kernel Error: %e\n\n', results(i).exponential_error);
end
% Tent map function
function x_next = tentMap(x)
    x_next = (x <= 0.5) .* (2 * x) + (x > 0.5) .* (2 * (1 - x));
end

% Generate Tent Map Orbit
function [orbit, full_sequence] = generateOrbit(x0, n_iterations)
    orbit = zeros(1, n_iterations + 1);
    orbit(1) = x0;
    
    full_sequence = zeros(1, n_iterations);
    
    for i = 1:n_iterations
        orbit(i+1) = tentMap(orbit(i));
        full_sequence(i) = orbit(i+1);
    end
end

% Gaussian Kernel
function K = gaussianKernel(x, x_train, sigma)
    [m, ~] = size(x);
    [n, ~] = size(x_train);
    
    K = zeros(m, n);
    for i = 1:m
        for j = 1:n
            K(i,j) = exp(-sum((x(i,:) - x_train(j,:)).^2) / (2 * sigma^2));
        end
    end
end

% Polynomial Kernel
function K = polynomialKernel(x, x_train, d, c)
    K = (x * x_train' + c).^d;
end

% Exponential Kernel
function K = exponentialKernel(x, x_train, sigma)
    [m, ~] = size(x);
    [n, ~] = size(x_train);
    
    K = zeros(m, n);
    for i = 1:m
        for j = 1:n
            K(i,j) = exp(-norm(x(i,:) - x_train(j,:)) / sigma);
        end
    end
end

% Train RBF Approximation
function [alpha, b] = trainRBFApproximation(x_train, y_train, kernel_func, sigma)
    % Compute kernel matrix
    K = kernel_func(x_train, x_train, sigma);
    
    % Solve for coefficients
    alpha = (K + 1e-10 * eye(size(K))) \ y_train;
    b = 0; % bias term (optional)
end

% Predict using RBF Approximation
function y_pred = predictRBF(x_test, x_train, y_train, kernel_func, alpha, sigma)
    % Compute kernel matrix between test and training points
    K_test = kernel_func(x_test, x_train, sigma);
    
    % Predict using learned coefficients
    y_pred = K_test * alpha;
end

% Compute Approximation Error
function error = computeApproximationError(y_true, y_pred)
    % Mean Squared Error
    error = mean((y_true - y_pred).^2);
end

% Run Tent Map Experiment
function results = runTentMapExperiment(initial_conditions, n_iterations, sigmas)
    % Preallocate results structure
    results = struct('initial_condition', {}, 'sigma', {}, ...
        'gaussian_error', {}, 'polynomial_error', {}, 'exponential_error', {});
    
    % Iterate through initial conditions
    for ic_idx = 1:length(initial_conditions)
        x0 = initial_conditions(ic_idx);
        
        % Generate orbit
        [orbit, full_sequence] = generateOrbit(x0, n_iterations);
        
        % Prepare training data
        x_train = orbit(1:end-2)';
        y_train = full_sequence(2:end)';
        
        % Iterate through sigma values
        for sig_idx = 1:length(sigmas)
            sigma = sigmas(sig_idx);
            
            % Gaussian Kernel Approximation
            [alpha_gauss, ~] = trainRBFApproximation(...
                x_train, y_train, @gaussianKernel, sigma);
            y_pred_gauss = predictRBF(...
                x_train, x_train, y_train, @gaussianKernel, alpha_gauss, sigma);
            gauss_error = computeApproximationError(y_train, y_pred_gauss);
            
            % Polynomial Kernel Approximation
            [alpha_poly, ~] = trainRBFApproximation(...
                x_train, y_train, @(x,x_t,sig) polynomialKernel(x,x_t,2,1), sigma);
            y_pred_poly = predictRBF(...
                x_train, x_train, y_train, @(x,x_t,sig) polynomialKernel(x,x_t,2,1), alpha_poly, sigma);
            poly_error = computeApproximationError(y_train, y_pred_poly);
            
            % Exponential Kernel Approximation
            [alpha_exp, ~] = trainRBFApproximation(...
                x_train, y_train, @exponentialKernel, sigma);
            y_pred_exp = predictRBF(...
                x_train, x_train, y_train, @exponentialKernel, alpha_exp, sigma);
            exp_error = computeApproximationError(y_train, y_pred_exp);
            
            % Store results
            results(end+1).initial_condition = x0;
            results(end).sigma = sigma;
            results(end).gaussian_error = gauss_error;
            results(end).polynomial_error = poly_error;
            results(end).exponential_error = exp_error;
        end
    end
end

% Visualization Function
function visualizeResults(results)
    % Create figure for error comparison
    figure('Position', [100, 100, 1000, 600]);
    
    % Subplot for Gaussian Kernel Errors
    subplot(1,3,1);
    gaussian_errors = [results.gaussian_error];
    initial_conditions = [results.initial_condition];
    boxplot(gaussian_errors, initial_conditions, 'Labels', arrayfun(@(x) num2str(x), initial_conditions, 'UniformOutput', false));
    title('Gaussian Kernel Errors');
    xlabel('Initial Condition');
    ylabel('Mean Squared Error');
    
    % Subplot for Polynomial Kernel Errors
    subplot(1,3,2);
    polynomial_errors = [results.polynomial_error];
    boxplot(polynomial_errors, initial_conditions, 'Labels', arrayfun(@(x) num2str(x), initial_conditions, 'UniformOutput', false));
    title('Polynomial Kernel Errors');
    xlabel('Initial Condition');
    ylabel('Mean Squared Error');
    
    % Subplot for Exponential Kernel Errors
    subplot(1,3,3);
    exponential_errors = [results.exponential_error];
    boxplot(exponential_errors, initial_conditions, 'Labels', arrayfun(@(x) num2str(x), initial_conditions, 'UniformOutput', false));
    title('Exponential Kernel Errors');
    xlabel('Initial Condition');
    ylabel('Mean Squared Error');
    
    % Save the figure
    saveas(gcf, 'tent_map_kernel_comparison.png');
end

