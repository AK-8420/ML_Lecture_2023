function x = solver_LASSO_cvx(data, para)
N = size(data.A, 2);

cvx_begin
variable x(N) complex
minimize( 1/2*sum(pow_abs(data.A*x - data.y, 2)) + para.lambda*sum(abs(x)) )
cvx_end