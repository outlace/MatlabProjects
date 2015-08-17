%{
f = @(t,x) [-x(1)+3*x(3);-x(2)+2*x(3);x(1)^2-2*x(3)];
[t,xa] = ode45(f,[0 1.5],[0 1/2 3]);
plot(t,xa(:,2))
title('y(t)')
xlabel('t'), ylabel('y')
%}

%v_rhs = '(V) - (V)**3/3 - (W) + I'
%w_rhs = 'a*((V) + b - c*(W))'
theta = [1.4 2.1];
yj = [2 3];
I = 0;
bias = 0;
for n = 1:2
f = @(t, x) [-x(1) + sigmoid(theta*yj' - bias) + I];
[t,xa] = ode45(f,[0 210],[1]);
yj(n) = xa;
plot(t, xa);
end