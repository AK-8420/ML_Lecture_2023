clear
close all

% detaset III
m = 20;
n = 40;
r = 2;
A = rand(m, r) * rand(r, n);
ninc = 100;
Q = randperm(m * n, ninc);
A(Q) = NaN;

figure
imagesc(A);

% setting
lambda = 1; % 0.5:0.5:3.0;
max_iteration = 1000;
stop_criteria = 1e-8;
eta = 0.5; % 1/2

% function
cost = @(Z, lambda) sum(abs(Z - A).^2, "omitnan") + lambda*NN(Z);
dp = @(Z) 2*fillmissing(Z - A,'constant',0);

% learning
for i = 1:length(lambda)
    % initializing
    Z = randn(m,n);
    converge_rate = zeros(1, max_iteration);

    for iter = 1:max_iteration
        Z_pre = Z;
        dpZ = dp(Z_pre);
        eta_t = eta;

        Z = prox(Z_pre - eta_t.*dpZ, lambda(i)*eta_t);
    
        converge_rate(iter) = sum(sum(abs(cost(Z, lambda(i)) - cost(Z_pre, lambda(i)))));
        % converge_rate(iter) = norm(Z(:) - Z_pre(:));
        if converge_rate(iter) < stop_criteria
            break;
        end
    end

    % view
    figure;
    semilogy(1:iter, converge_rate(1:iter));
    ylabel("$\| f(Z^{(t)}) - f(\hat{Z}) \|_1$", 'Interpreter','latex')
    xlabel("iteration")

    figure
    imagesc(Z);
end

save("problem5_result")


function value = NN(X)
    [~,S,~] = svd(X);
    value = sum(diag(S));
end
function X = prox(X, gamma)
    [U,S,V] = svd(X);
    X = U*max(S - gamma, 0)*V';
end