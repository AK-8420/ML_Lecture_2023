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
[w_N, converge_rate_N, idx_N] = problem1_Newton(data);

% Generate test data (same as dataset IV)
nt = 100;
xt = 3 * (rand(nt, 4) - 0.5);
yt = (2 * xt(:, 1) - 1 * xt(:,2) + 0.5 + 0.5 * randn(nt, 1)) > 0;
yt = 2 * yt -1;

xt = cat(2, xt, ones(nt,1));

% evaluate
f = @(w) 2*(xt*w > 0) - 1;
correct_number_GD = sum( yt == f(w_GD) )
correct_number_Newton = sum( yt == f(w_N) )
difference_of_w = norm(w_GD - w_N)

% view (all)
f1 = figure;
semilogy(1:idx_GD, converge_rate_GD(1:idx_GD));
hold on
semilogy(1:idx_N, converge_rate_N(1:idx_N));
hold off
legend("Steepest gradient descent method", "Newton method")
ylabel("$\| J(w^{(t)}) - J(\hat{w}) \|_1$", 'Interpreter','latex')
xlabel("iteration")
ylim([1e-8, 1e2])
f1.Position(3:4) = [480 320];

% view (100 iter)
f2 = figure;
range = 100;
semilogy(1:range, converge_rate_GD(1:range));
hold on
semilogy(1:range, converge_rate_N(1:range));
hold off
legend("Steepest gradient descent method", "Newton method")
ylabel("$\| J(w^{(t)}) - J(\hat{w}) \|_1$", 'Interpreter','latex')
xlabel("iteration")
ylim([1e-8, 1e2])
f2.Position(3:4) = [480 320];

% save
print('-f1', "problem1_result",'-dpng')
print('-f2', "problem1_result_100iter",'-dpng')
clear("f1")
clear("f2")
save("problem1_result")