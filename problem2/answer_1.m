clear
close all

% deta
data.A = [3, 0.5;
         0.5, 1];
data.mu = [1; 2];
data.max_iteration = 1000;

lambda = 0.01:0.01:10.0;
len = length(lambda);

% learning
w = zeros(2, len);
converge_rate = zeros(len, data.max_iteration);
idx = zeros(1, len);

for i = 1:len
    data.lambda = lambda(i);
    [w(:, i), converge_rate(i, :), idx(i)] = Standard_PG(data);
end

% view
f1 = figure;
semilogy(1:idx(1), converge_rate(1, 1:idx(1)));
ylabel("$\| J(w^{(t)}) - J(\hat{w}) \|_1$", 'Interpreter','latex')
xlabel("Iteration")
f1.Position(3:4) = [480 320];

f2 = figure;
plot3(lambda, w(1,:), w(2,:));
grid on
xlabel("$\lambda$", 'Interpreter','latex')
ylabel("$w_1$", 'Interpreter','latex')
zlabel("$w_2$", 'Interpreter','latex')
f2.Position(3:4) = [480 320];

% save
print('-f1', "problem2_1_result",'-dpng')
print('-f2', "problem2_1_result_optimal",'-dpng')
clear("f1")
clear("f2")
save("problem2_1_result")