clear
close all

% detaset IV
n = 200;
x = 3 * (rand(n, 4) - 0.5);
y = (2 * x(:, 1) - 1 * x(:,2) + 0.5 + 0.5 * randn(n, 1)) > 0;
y = 2 * y -1;

data.n = n;
data.x = cat(2, x, ones(n,1)); % add 1
data.y = y;
data.lambda = 0.25;

% learning
[w_GD, converge_rate_GD, idx_GD] = problem1_GD(data);

% Generate test data (same as dataset IV)
nt = 100;
xt = 3 * (rand(nt, 4) - 0.5);
yt = (2 * xt(:, 1) - 1 * xt(:,2) + 0.5 + 0.5 * randn(nt, 1)) > 0;
yt = 2 * yt -1;

xt = cat(2, xt, ones(nt,1));

% evaluate
f = @(w) 2*(xt*w > 0) - 1;
correct_number = sum( yt == f(w_GD) )

% view
figure
hold on
semilogy(1:idx_GD, converge_rate_GD(1:idx_GD));
hold off