addpath(genpath("cvx"));

% cvx_setup; % MATLAB起動の度に実行

% settings
A = [3, 0.5;
     0.5, 1];
mu = [1; 2];

lambda = 0.01:0.01:10.0;
len = length(lambda);

% learning
w_hat = zeros(2, len);
for i = 1:len
    cvx_begin
    variables w(2)
    minimize( 1/2*((w - mu)')*A*(w - mu) + lambda(i)*sum(abs(w)) )
    cvx_end

    % check optimization status
    if ~strcmp(cvx_status, 'Solved')
        error("CVX Failed");
    end
    w_hat(:, i) = w;
end

f1 = figure;
plot3(lambda, w_hat(1,:), w_hat(2,:));
grid on
xlabel("$\lambda$", 'Interpreter','latex')
ylabel("$w_1$", 'Interpreter','latex')
zlabel("$w_2$", 'Interpreter','latex')
f1.Position(3:4) = [480 320];