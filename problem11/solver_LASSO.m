function x = solver_LASSO(data, para)
%----------------------------
% RMSProp
%----------------------------
% training setting
eta = 0.1;
rho = 0.9;

% function
dp = @(x) data.A'*(data.A*x - data.y);
% ST = @(x, gamma) sign(x).*max(abs(x)-gamma, 0); % Varid for only real number

% initializing
N = size(data.A'*data.y, 1);
x = zeros(N, 1);
v = zeros(N, 1);

% main loop
converge_rate = zeros(1, para.max_iteration);

for i = 1:para.max_iteration
    x_pre = x;
    dpw = dp(x_pre);

    v = rho*v + (1 - rho)*(dpw.^2);
    eta_t = eta./(sqrt(v) + eps);

    % Soft Thresholding
    % x = ST(x_pre - eta_t.*dpw, para.lambda*eta_t);
    x = x_pre - eta_t.*dpw;
    gamma = para.lambda*eta_t;
    for j = 1:N
        if x(j) > gamma
            x(j) = x(j) - gamma;
        elseif x(j) < -gamma
            x(j) = x(j) + gamma;
        else
            x(j) = 0;
        end
    end

    converge_rate(i) = norm(x - x_pre);

    if converge_rate(i) < para.stop_criteria
        break;
    end
end