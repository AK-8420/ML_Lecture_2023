clear
close all

% deta
data.A = [3, 0.5;
         0.5, 1];
data.mu = [1; 2];
data.max_iteration = 1000;

lambda = 0.01:0.01:10.0;
len = length(lambda);

w = zeros(2, len);
optimal_cost = zeros(1, len);

% learning
for i = 1:len
    data.lambda = lambda(i);
    [w(:, i), converge_rate, idx] = Standard_PG(data);

    optimal_cost(i) = converge_rate(idx);
end

% view
f1 = figure;
semilogy(lambda, optimal_cost);
ylabel("$\| J(w^{(t)}) - J(\hat{w}) \|_1$", 'Interpreter','latex')
xlabel("$\lambda$", 'Interpreter','latex')
ylim([1e-8, 1e2])
f1.Position(3:4) = [480 320];

% save
print('-f1', "problem2_1_result",'-dpng')
clear("f1")
save("problem2_1_result")