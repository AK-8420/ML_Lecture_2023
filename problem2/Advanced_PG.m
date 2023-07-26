function [w, converge_rate, i] = Advanced_PG(data)
% Adam (RMSProp × Momentum)

% training setting
stop_criteria = 1e-8;
eta = 1/max(eig(data.A));

% function
J = @(w) 1/2*((w - data.mu)')*data.A*(w - data.mu) + data.lambda*sum(abs(w));
dp = @(w) 1/2*(data.A + data.A')*(w - data.mu);
ST = @(x, gamma) sign(x).*max(abs(x)-gamma, 0);

% initializing
w = [3; -1];

% main loop
converge_rate = zeros(1, data.max_iteration);

for i = 1:data.max_iteration
    w_pre = w;
    w = ST(w_pre - eta*dp(w_pre), data.lambda*eta);

    converge_rate(i) = abs(J(w_pre) - J(w));

    if converge_rate(i) < stop_criteria
        break;
    end
end