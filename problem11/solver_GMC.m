function x = solver_GMC(data, para)
%% Settings
%-----------------------------------------------------
% step size
%-----------------------------------------------------
% rho = max{1, gamma/(1 - gamma)}*|| A^T*A ||_2
% A^T*A = I (NxN matrix)
% || A^T*A ||_2 = sqrt( N )
AtA = data.A'*data.A;
rho = max(1, para.gamma/(1-para.gamma))*norm( AtA(:) );

% 0 < mu < 2/rho
mu = max(eps, 2/rho - 0.01);

%-----------------------------------------------------
% proximity operator
%-----------------------------------------------------
% Soft Thresholfing
ST = @(x, gamma) sign(x).*max(abs(x)-gamma, 0);


%-----------------------------------------------------
% Initial values
%-----------------------------------------------------
N = size(data.A, 2);
x = zeros(N, 1);
v = zeros(N, 1);


%% main
%-----------------------------------------------------
% main loop
%-----------------------------------------------------
converge_rate = zeros(1, para.max_iteration);
for i = 1:para.max_iteration
    x_pre = x;

    % update
    w = x - mu*(data.A')*( data.A*(x + para.gamma*(v - x)) - data.y );
    u = v - mu*para.gamma*AtA*( v - x );
    x = ST(w, mu*para.lambda);
    v = ST(u, mu*para.lambda);

    % check convergence
    converge_rate(i) = norm(x - x_pre);
    % fprintf('iter: %d, Error(X) = %f\n', i, converge_rate(i));

    % exit loop
    if converge_rate(i) < para.stop_criteria
        break;
    end
end