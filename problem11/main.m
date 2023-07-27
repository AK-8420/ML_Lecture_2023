%==================================
% GMC penalty vs. L1
% based on Sparse_Regularization_via_Convex_Analysis
%
% 2023.7.27
% Author: Akari Katsuma (katsuma.a.aa@m.titech.ac.jp)
%==================================
clear
close all

%----------------------------------------
% training setting
%----------------------------------------
para.stop_criteria = 1e-8;
para.max_iteration = 10000;
repeat_number = 10;   % the count of experiments

noise_strength = 1.0; % standard deviation of gaussian noise
lambda_all = 0.5:0.25:3.5;

%----------------------------------------
% data
%----------------------------------------
M = 100;
N = 256;
F1 = 0.1;
F2 = 0.22;

% true signal
t = 0:M-1;
g = (2*cos(2*pi*F1*t) + sin(2*pi*F2*t))';

% inverse fourier transform matrix
data.A = zeros(M, N);
for m = 1:M
    for n = 1:N
        data.A(m,n) = exp(1i*2*pi/N*(m-1)*(n-1)) / sqrt(N);
    end
end

%----------------------------------------
% training
%----------------------------------------
x_L1 = zeros(N, length(lambda_all), repeat_number);
x_GMC = zeros(N, length(lambda_all), repeat_number);
% x_L1_cvx = zeros(N, length(lambda_all), repeat_number);
ys = zeros(M, length(lambda_all), repeat_number);

for i = 1:length(lambda_all)
    para.lambda = lambda_all(i);
    fprintf("Now processing (lambda = %f)\n", para.lambda)

    for j = 1:repeat_number
        fprintf("%d, ", j);

        % observed signal (with random noise)
        noise = noise_strength*randn(M, 1);
        data.y = g + noise;

        % % get solution for varidation
        % x_L1_cvx(:, i, j) = solver_LASSO_cvx(data, para);

        % get solution
        para.gamma = 0;
        x_L1(:, i, j) = solver_GMC(data, para);
        para.gamma = 0.8;
        x_GMC(:, i, j) = solver_GMC(data, para);
    
        % save
        ys(:, i, j) = data.y;
    end
    fprintf("\n");
end
%----------------------------------------
% Avaraged RMSE
%----------------------------------------
RMSE_L1 = zeros(length(lambda_all), repeat_number); % for each lambda
RMSE_GMC = zeros(length(lambda_all), repeat_number); % for each lambda
% RMSE_L1_cvx = zeros(length(lambda_all), repeat_number); % for each lambda
for i = 1:length(lambda_all)
    for j = 1:repeat_number
        RMSE_L1(i, j) = norm(g - data.A*x_L1(:,i,j)) / sqrt(M);
        RMSE_GMC(i, j) = norm(g - data.A*x_GMC(:,i,j)) / sqrt(M);
        % RMSE_L1_cvx(i, j) = norm(g - data.A*x_L1_cvx(:,i,j)) / sqrt(M);
    end
end

ARMSE_L1 = mean(RMSE_L1, 2);
ARMSE_GMC = mean(RMSE_GMC, 2);
% ARMSE_L1_cvx = mean(RMSE_L1_cvx, 2)

[~, optimal_idx] = min(ARMSE_GMC);
fprintf("The best lambda for GMC is %f)\n", lambda_all(optimal_idx));
ARMSE_L1(optimal_idx)
ARMSE_GMC(optimal_idx)


% %----------------------------------------
% % Varidation
% %----------------------------------------
% Numel = N*length(lambda_all)*repeat_number;
% difference = norm(reshape(x_L1_cvx, [Numel, 1]) - reshape(x_L1, [Numel, 1]));


%----------------------------------------
% view
%----------------------------------------
% original signal
f1 = figure;
subplot(121), plot(1:M, g), title('True signal');
ylim([-5, 5]);
xlabel("m");
subplot(122), plot(1:M, ys(:,1,1)), title('Observed signal');
ylim([-5, 5]);
xlabel("m");
f1.Position(3:4) = [640 160];

% Fourier coefficients of original signal
f = linspace(0, (N/2)/N, N/2);
fftg = abs(data.A'*g);
ffty = abs(data.A'*ys(:,1,1));
f2 = figure;
subplot(121), stem(f, fftg(1:(N/2)), "MarkerSize", 3), title('FFT of True signal');
ylim([0, max(ffty)+0.2]);
xlabel("Frequency");
subplot(122), stem(f, ffty(1:(N/2)), "MarkerSize", 3), title('FFT of Noisy signal');
ylim([0, max(ffty)+0.2]);
xlabel("Frequency");
f2.Position(3:4) = [640 160];

% Denoised signal
f3 = figure;
subplot(121), plot(1:M, data.A*x_L1(:, optimal_idx, 1)), title('Denoised [L1 norm]');
ylim([-5, 5]);
xlabel("m");
subplot(122), plot(1:M, data.A*x_GMC(:, optimal_idx, 1)), title('Denoised [GMC penalty]');
ylim([-5, 5]);
xlabel("m");
f3.Position(3:4) = [640 160];

% Fourier coefficients of Denoised signals
f = linspace(0, (N/2)/N, N/2);
fftg = abs(x_L1(:, optimal_idx, 1));
ffty = abs(x_GMC(:, optimal_idx, 1));
f4 = figure;
subplot(121), stem(f, fftg(1:(N/2)), "MarkerSize", 3), title('FFT [L1 norm]');
ylim([0, max(ffty)+0.2]);
xlabel("Frequency");
subplot(122), stem(f, ffty(1:(N/2)), "MarkerSize", 3), title('FFT [GMC penalty]');
ylim([0, max(ffty)+0.2]);
xlabel("Frequency");
f4.Position(3:4) = [640 160];

% ARMSE
f5 = figure;
plot(lambda_all, ARMSE_L1);
hold on
plot(lambda_all, ARMSE_GMC);
hold off
legend("L1 norm", "GMC penalty");
ylabel("Avarage RMSE", "Interpreter", "latex");
xlabel("$\lambda$", "Interpreter", "latex");
f5.Position(3:4) = [640 320];

% compare
f6 = figure;
range = 1:M/4;
plot(range, g(range), "k");
hold on;
opt_L1 = data.A*x_L1(:, optimal_idx, 1);
opt_GMC = data.A*x_GMC(:, optimal_idx, 1);
plot(range, opt_L1(range), "b--");
plot(range, opt_GMC(range), "r-.");
hold off;
ylim([-5, 5]);
xlabel("m");
legend("true signal", "L1 norm", "GMC penalty");


%----------------------------------------
% save
%----------------------------------------
print('-f1', "true_signal_and_noisy_signal",'-dpng')
print('-f2', "true_signal_and_noisy_signal_FFT",'-dpng')
print('-f3', "result_time_domain",'-dpng')
print('-f4', "result_frequency_domain",'-dpng')
print('-f5', "result_ARMSE",'-dpng')
print('-f6', "result_compare_signals",'-dpng')
clear("f1")
clear("f2")
clear("f3")
clear("f4")
clear("f5")
clear("f6")
save("problem11_result")