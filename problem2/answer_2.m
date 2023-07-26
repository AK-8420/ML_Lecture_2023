clear
close all

% deta
data.A = [300, 0.5;
          0.5, 10];
data.mu = [1; 2];
data.lambda = 0.25;

% learning
[w_1, converge_rate_1, idx_1] = Standard_PG(data);
[w_2, converge_rate_2, idx_2] = Advanced_PG(data);

% view (all)
f1 = figure;
semilogy(1:idx_1, converge_rate_1(1:idx_1));
hold on
semilogy(1:idx_2, converge_rate_2(1:idx_2));
hold off
legend("Standard PG", "Advanced PG")
ylabel("$\| J(w^{(t)}) - J(\hat{w}) \|_1$", 'Interpreter','latex')
xlabel("iteration")
ylim([1e-8, 1e2])
f1.Position(3:4) = [480 320];

% save
print('-f1', "problem2_2_result",'-dpng')
clear("f1")
save("problem2_2_result")