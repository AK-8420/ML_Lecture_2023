function x = solver_LASSO_cvx(data, para)

N = size(data.A'*data.y, 1);
x = zeros(N, 1);

cvx_begin
variables x(N,1)
minimize( 1/2*sum((real(data.A*x) - data.y).^2) + para.lambda*sum(abs(x)) )
cvx_end

% check optimization status
if ~strcmp(cvx_status, 'Solved')
    error("CVX Failed");
end