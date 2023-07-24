function [w, converge_rate, i] = problem1_Newton(data)

% training setting
stop_criteria = 1e-8;
max_iteration = 10000;
alpha = 1;

% initializing
w = zeros(size(data.x, 2), 1);

% main loop
converge_rate = zeros(1, max_iteration);

for i = 1:max_iteration
    w_pre = w;
    w = w_pre - alpha*(dJ(w_pre, data)./d2J(w_pre, data));

    converge_rate(i) = abs(J(w_pre, data) - J(w, data));

    if converge_rate(i) < stop_criteria
        break;
    end
end

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

% Hessian
function d = d2J(w, data)
    temp = zeros(size(data.x, 2), data.n);
    for j = 1:data.n
      temp(:, j) = (-data.y(j)*data.x(j,:)').^2*(exp(-data.y(j)*(w')*(data.x(j,:)')))/(( 1 + exp(-data.y(j)*(w')*(data.x(j,:)')) ).^2);
    end
    d = sum(temp, 2) + 2*data.lambda;
end
%------------------------------

end