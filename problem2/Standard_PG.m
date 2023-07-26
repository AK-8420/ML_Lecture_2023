function [w, converge_rate, i] = Standard_PG(data)

% training setting
stop_criteria = 1e-8;
eta = 1/max(eig(data.A));
gamma = 1/eta - 0.01;   % gamma^(-1) >= eta

% function
J = @(w) 1/2*((w - data.mu)')*data.A*(w - data.mu) + data.lambda*sum(abs(w));
dp = @(w) 1/2*(data.A + data.A')*(w - data.mu);
ST = @(x, gamma) sign(x).*max(abs(x)-gamma, 0);

% initializing
w = randn(size(data.mu));

% main loop
converge_rate = zeros(1, data.max_iteration);

for i = 1:data.max_iteration
    w_pre = w;
    w = ST(w_pre - eta*dp(w_pre), gamma);

    converge_rate(i) = abs(J(w_pre) - J(w));

    if converge_rate(i) < stop_criteria
        break;
    end
end