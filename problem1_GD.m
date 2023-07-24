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

% training setting
stop_criteria = 1e-8;
max_iteration = 10000;
alpha = 0.01;

% initializing
w = zeros(size(data.x, 2), 1);

% main loop
converge_rate = zeros(1, max_iteration);

for i = 1:max_iteration
    w_pre = w;
    w = w_pre - alpha*dJ(w_pre, data);

    converge_rate(i) = abs(J(w_pre, data) - J(w, data));

    if converge_rate(i) < stop_criteria
        break;
    end
end

% show result
w

% test data
nt = 100;
xt = 3 * (rand(nt, 4) - 0.5);
yt = (2 * xt(:, 1) - 1 * xt(:,2) + 0.5 + 0.5 * randn(nt, 1)) > 0;
yt = 2 * yt -1;

% evaluate
xt = cat(2, xt, ones(nt,1));
f = 2*(xt*w > 0) - 1;
correct_number = sum( yt == f )

% view
progressWindow = figure;
semilogy(1:i, converge_rate(1:i));

%------------------------------
% cost function
function cost = J(w, data)
    temp = zeros(1,data.n);
    for j = 1:data.n
      temp(j) = log( 1 + exp(-data.y(j)*(w')*(data.x(j,:)')) );
    end
    cost = sum(temp) + data.lambda*(w')*w;
end

% differentiation
function d = dJ(w, data)
    temp = zeros(size(data.x, 2), data.n);
    for j = 1:data.n
      temp(:, j) = (-data.y(j)*data.x(j,:)')*(exp(-data.y(j)*(w')*(data.x(j,:)')))/( 1 + exp(-data.y(j)*(w')*(data.x(j,:)')) );
    end
    d = sum(temp, 2) + 2*data.lambda*w;
end